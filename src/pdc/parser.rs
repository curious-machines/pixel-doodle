use super::ast::*;
use super::error::PdcError;
use super::span::{IdAlloc, Span, Spanned};
use super::token::{Token, TokenKind};

pub fn parse(tokens: Vec<Token>) -> Result<Program, PdcError> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    ids: IdAlloc,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            ids: IdAlloc::new(),
        }
    }

    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        self.pos += 1;
        tok
    }

    fn at(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(kind)
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Span, PdcError> {
        if self.at(kind) {
            let span = self.span();
            self.advance();
            Ok(span)
        } else {
            Err(PdcError::Parse {
                span: self.span(),
                message: format!("expected {:?}, got {:?}", kind, self.peek()),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), PdcError> {
        if let TokenKind::Ident(name) = self.peek().clone() {
            let span = self.span();
            self.advance();
            Ok((name, span))
        } else {
            Err(PdcError::Parse {
                span: self.span(),
                message: format!("expected identifier, got {:?}", self.peek()),
            })
        }
    }

    fn parse_program(&mut self) -> Result<Program, PdcError> {
        let mut items = Vec::new();
        while *self.peek() != TokenKind::Eof {
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Spanned<Item>, PdcError> {
        let start = self.span();
        match self.peek().clone() {
            TokenKind::Builtin => self.parse_builtin_decl(start),
            TokenKind::Const => self.parse_const_decl(start),
            TokenKind::Var => self.parse_var_decl(start),
            _ => {
                // Expression statement or assignment
                let expr = self.parse_expr()?;
                // Check for assignment: `name = expr`
                if *self.peek() == TokenKind::Eq {
                    if let Expr::Variable(name) = &expr.node {
                        let name = name.clone();
                        self.advance(); // consume '='
                        let value = self.parse_expr()?;
                        let span = Span::new(start.start, value.span.end);
                        return Ok(self.ids.spanned(Item::Assign { name, value }, span));
                    }
                }
                let span = Span::new(start.start, expr.span.end);
                Ok(self.ids.spanned(Item::ExprStmt(expr), span))
            }
        }
    }

    fn parse_builtin_decl(&mut self, start: Span) -> Result<Spanned<Item>, PdcError> {
        self.advance(); // consume 'builtin'
        self.expect(&TokenKind::Const)?;
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
        Ok(self.ids.spanned(Item::BuiltinDecl { name, ty }, span))
    }

    fn parse_const_decl(&mut self, start: Span) -> Result<Spanned<Item>, PdcError> {
        self.advance(); // consume 'const'
        let (name, _) = self.expect_ident()?;
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        let span = Span::new(start.start, value.span.end);
        Ok(self.ids.spanned(Item::ConstDecl { name, ty, value }, span))
    }

    fn parse_var_decl(&mut self, start: Span) -> Result<Spanned<Item>, PdcError> {
        self.advance(); // consume 'var'
        let (name, _) = self.expect_ident()?;
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        let span = Span::new(start.start, value.span.end);
        Ok(self.ids.spanned(Item::VarDecl { name, ty, value }, span))
    }

    fn parse_type(&mut self) -> Result<PdcType, PdcError> {
        if let TokenKind::Ident(name) = self.peek().clone() {
            let ty = match name.as_str() {
                "f32" => PdcType::F32,
                "f64" => PdcType::F64,
                "i32" => PdcType::I32,
                "u32" => PdcType::U32,
                "bool" => PdcType::Bool,
                "Path" => PdcType::PathHandle,
                _ => {
                    return Err(PdcError::Parse {
                        span: self.span(),
                        message: format!("unknown type '{name}'"),
                    })
                }
            };
            self.advance();
            Ok(ty)
        } else {
            Err(PdcError::Parse {
                span: self.span(),
                message: format!("expected type name, got {:?}", self.peek()),
            })
        }
    }

    // ── Expression parsing (precedence climbing) ──

    fn parse_expr(&mut self) -> Result<Spanned<Expr>, PdcError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_and()?;
        while *self.peek() == TokenKind::PipePipe {
            self.advance();
            let right = self.parse_and()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op: BinOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_comparison()?;
        while *self.peek() == TokenKind::AmpAmp {
            self.advance();
            let right = self.parse_comparison()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op: BinOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_addition()?;
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::BangEq => BinOp::NotEq,
                TokenKind::Lt => BinOp::Lt,
                TokenKind::LtEq => BinOp::LtEq,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::GtEq => BinOp::GtEq,
                _ => break,
            };
            self.advance();
            let right = self.parse_addition()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_addition(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_multiplication()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplication()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let start = self.span();
        match self.peek() {
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                let span = Span::new(start.start, operand.span.end);
                Ok(self.ids.spanned(
                    Expr::UnaryOp {
                        op: UnaryOp::Neg,
                        operand: Box::new(operand),
                    },
                    span,
                ))
            }
            TokenKind::Bang => {
                self.advance();
                let operand = self.parse_unary()?;
                let span = Span::new(start.start, operand.span.end);
                Ok(self.ids.spanned(
                    Expr::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    },
                    span,
                ))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.advance(); // consume '.'
                    let (method, _) = self.expect_ident()?;
                    if *self.peek() == TokenKind::LParen {
                        // Method call: expr.method(args)
                        self.advance(); // consume '('
                        let args = self.parse_call_args()?;
                        let span = Span::new(
                            expr.span.start,
                            self.tokens[self.pos - 1].span.end,
                        );
                        expr = self.ids.spanned(
                            Expr::MethodCall {
                                object: Box::new(expr),
                                method,
                                args,
                            },
                            span,
                        );
                    } else {
                        // Field access (deferred — treat as method call with no args for now)
                        let span = Span::new(
                            expr.span.start,
                            self.tokens[self.pos - 1].span.end,
                        );
                        expr = self.ids.spanned(
                            Expr::MethodCall {
                                object: Box::new(expr),
                                method,
                                args: Vec::new(),
                            },
                            span,
                        );
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let start = self.span();
        match self.peek().clone() {
            TokenKind::IntLit(val) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::Int(val)), start))
            }
            TokenKind::FloatLit(val) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::Float(val)), start))
            }
            TokenKind::BoolLit(val) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::Bool(val)), start))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Check for function/constructor call: name(args)
                if *self.peek() == TokenKind::LParen {
                    self.advance(); // consume '('
                    let args = self.parse_call_args()?;
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    Ok(self.ids.spanned(Expr::Call { name, args }, span))
                } else {
                    Ok(self.ids.spanned(Expr::Variable(name), start))
                }
            }
            TokenKind::LParen => {
                self.advance(); // consume '('
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            _ => Err(PdcError::Parse {
                span: start,
                message: format!("unexpected token {:?}", self.peek()),
            }),
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<Spanned<Expr>>, PdcError> {
        let mut args = Vec::new();
        if *self.peek() != TokenKind::RParen {
            args.push(self.parse_expr()?);
            while *self.peek() == TokenKind::Comma {
                self.advance();
                if *self.peek() == TokenKind::RParen {
                    break; // trailing comma
                }
                args.push(self.parse_expr()?);
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(args)
    }
}
