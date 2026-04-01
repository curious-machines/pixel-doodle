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
        let mut stmts = Vec::new();
        while *self.peek() != TokenKind::Eof {
            stmts.push(self.parse_stmt()?);
        }
        Ok(Program { stmts })
    }

    // ── Statement parsing ──

    fn parse_stmt(&mut self) -> Result<Spanned<Stmt>, PdcError> {
        let start = self.span();
        match self.peek().clone() {
            TokenKind::Builtin => self.parse_builtin_decl(start),
            TokenKind::Const => self.parse_const_decl(start),
            TokenKind::Var => self.parse_var_decl(start),
            TokenKind::If => self.parse_if(start),
            TokenKind::While => self.parse_while(start),
            TokenKind::For => self.parse_for(start),
            TokenKind::Loop => self.parse_loop(start),
            TokenKind::Match => self.parse_match(start),
            TokenKind::Break => {
                self.advance();
                Ok(self.ids.spanned(Stmt::Break, start))
            }
            TokenKind::Continue => {
                self.advance();
                Ok(self.ids.spanned(Stmt::Continue, start))
            }
            TokenKind::Return => self.parse_return(start),
            TokenKind::Import => self.parse_import(start),
            TokenKind::Struct => self.parse_struct_def(start),
            TokenKind::Enum => self.parse_enum_def(start),
            TokenKind::Fn => self.parse_fn_def(start),
            _ => {
                // Expression statement or assignment
                let expr = self.parse_expr()?;

                // Simple assignment: name = expr
                if *self.peek() == TokenKind::Eq {
                    if let Expr::Variable(name) = &expr.node {
                        let name = name.clone();
                        self.advance();
                        let value = self.parse_expr()?;
                        let span = Span::new(start.start, value.span.end);
                        return Ok(self.ids.spanned(Stmt::Assign { name, value }, span));
                    }
                }

                // Compound assignment: name += expr → name = name + expr
                let compound_op = match self.peek() {
                    TokenKind::PlusEq => Some(BinOp::Add),
                    TokenKind::MinusEq => Some(BinOp::Sub),
                    TokenKind::StarEq => Some(BinOp::Mul),
                    TokenKind::SlashEq => Some(BinOp::Div),
                    TokenKind::PercentEq => Some(BinOp::Mod),
                    _ => None,
                };
                if let Some(op) = compound_op {
                    if let Expr::Variable(name) = &expr.node {
                        let name = name.clone();
                        self.advance(); // consume the compound operator
                        let rhs = self.parse_expr()?;
                        let rhs_span = rhs.span;
                        // Desugar: x += e → x = x + e
                        let var_ref = self.ids.spanned(Expr::Variable(name.clone()), expr.span);
                        let binop = self.ids.spanned(
                            Expr::BinaryOp {
                                op,
                                left: Box::new(var_ref),
                                right: Box::new(rhs),
                            },
                            Span::new(expr.span.start, rhs_span.end),
                        );
                        let span = Span::new(start.start, rhs_span.end);
                        return Ok(self.ids.spanned(Stmt::Assign { name, value: binop }, span));
                    }
                }

                let span = Span::new(start.start, expr.span.end);
                Ok(self.ids.spanned(Stmt::ExprStmt(expr), span))
            }
        }
    }

    fn parse_builtin_decl(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance();
        self.expect(&TokenKind::Const)?;
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
        Ok(self.ids.spanned(Stmt::BuiltinDecl { name, ty }, span))
    }

    fn parse_const_decl(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'const'
        // Tuple destructuring: const (a, b, c) = expr
        if *self.peek() == TokenKind::LParen {
            return self.parse_tuple_destructure(start, true);
        }
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
        Ok(self.ids.spanned(Stmt::ConstDecl { name, ty, value }, span))
    }

    fn parse_var_decl(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'var'
        // Tuple destructuring: var (a, b, c) = expr
        if *self.peek() == TokenKind::LParen {
            return self.parse_tuple_destructure(start, false);
        }
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
        Ok(self.ids.spanned(Stmt::VarDecl { name, ty, value }, span))
    }

    fn parse_tuple_destructure(&mut self, start: Span, is_const: bool) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume '('
        let mut names = Vec::new();
        if !self.at(&TokenKind::RParen) {
            loop {
                let (name, _) = self.expect_ident()?;
                names.push(name);
                if *self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        let span = Span::new(start.start, value.span.end);
        Ok(self.ids.spanned(Stmt::TupleDestructure { names, value, is_const }, span))
    }

    fn parse_if(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'if'
        let condition = self.parse_expr()?;
        let then_body = self.parse_block()?;

        let mut elsif_clauses = Vec::new();
        while *self.peek() == TokenKind::Elsif {
            self.advance();
            let cond = self.parse_expr()?;
            let body = self.parse_block()?;
            elsif_clauses.push((cond, body));
        }

        let else_body = if *self.peek() == TokenKind::Else {
            self.advance();
            Some(self.parse_block()?)
        } else {
            None
        };

        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(
            Stmt::If {
                condition,
                then_body,
                elsif_clauses,
                else_body,
            },
            span,
        ))
    }

    fn parse_while(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'while'
        let condition = self.parse_expr()?;
        let body = self.parse_block()?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(Stmt::While { condition, body }, span))
    }

    fn parse_for(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'for'
        let (var_name, _) = self.expect_ident()?;
        self.expect(&TokenKind::In)?;
        let range_start = self.parse_expr()?;
        self.expect(&TokenKind::DotDot)?;
        let range_end = self.parse_expr()?;
        let body = self.parse_block()?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(
            Stmt::For {
                var_name,
                start: range_start,
                end: range_end,
                body,
            },
            span,
        ))
    }

    fn parse_loop(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'loop'
        let body = self.parse_block()?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(Stmt::Loop { body }, span))
    }

    fn parse_return(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'return'
        // Return value is optional — check if next token could start an expression
        let value = match self.peek() {
            TokenKind::RBrace | TokenKind::Eof => None,
            _ => Some(self.parse_expr()?),
        };
        let end = if let Some(ref v) = value {
            v.span.end
        } else {
            start.end
        };
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(Stmt::Return(value), span))
    }

    fn parse_fn_def(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'fn'
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;

        // Parse parameters
        let mut params = Vec::new();
        if !self.at(&TokenKind::RParen) {
            loop {
                let (pname, _) = self.expect_ident()?;
                self.expect(&TokenKind::Colon)?;
                let pty = self.parse_type()?;
                params.push(Param { name: pname, ty: pty });
                if *self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;

        // Return type
        let return_type = if *self.peek() == TokenKind::Arrow {
            self.advance();
            self.parse_type()?
        } else {
            PdcType::Void
        };

        let body = self.parse_block()?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(
            Stmt::FnDef(FnDef {
                name,
                params,
                return_type,
                body,
            }),
            span,
        ))
    }

    fn parse_struct_def(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'struct'
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let (fname, _) = self.expect_ident()?;
            self.expect(&TokenKind::Colon)?;
            let fty = self.parse_type()?;
            fields.push(StructField { name: fname, ty: fty });
            if *self.peek() == TokenKind::Comma {
                self.advance();
            }
        }
        self.expect(&TokenKind::RBrace)?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(
            Stmt::StructDef(StructDef { name, fields }),
            span,
        ))
    }

    fn parse_match(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'match'
        let scrutinee = self.parse_expr()?;
        self.expect(&TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let pattern = self.parse_match_pattern()?;
            self.expect(&TokenKind::FatArrow)?; // =>
            let body = self.parse_block()?;
            // Optional comma after arm
            if *self.peek() == TokenKind::Comma {
                self.advance();
            }
            arms.push(MatchArm { pattern, body });
        }
        self.expect(&TokenKind::RBrace)?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(Stmt::Match { scrutinee, arms }, span))
    }

    fn parse_match_pattern(&mut self) -> Result<MatchPattern, PdcError> {
        // Catch-all wildcard: _
        if let TokenKind::Ident(name) = self.peek().clone() {
            if name == "_" {
                self.advance();
                return Ok(MatchPattern::Wildcard);
            }
        }

        // Dot-prefix shorthand: .Variant or .Variant(a, b, c)
        // The enum_name will be filled in by the type checker from the scrutinee type.
        let enum_name = if *self.peek() == TokenKind::Dot {
            self.advance();
            String::new() // empty = infer from scrutinee
        } else {
            // Full syntax: EnumName.Variant
            let (name, _) = self.expect_ident()?;
            self.expect(&TokenKind::Dot)?;
            name
        };

        let (variant, _) = self.expect_ident()?;

        // Optional destructuring bindings
        let bindings = if *self.peek() == TokenKind::LParen {
            self.advance();
            let mut binds = Vec::new();
            if !self.at(&TokenKind::RParen) {
                loop {
                    let (bname, _) = self.expect_ident()?;
                    binds.push(bname);
                    if *self.peek() == TokenKind::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RParen)?;
            binds
        } else {
            Vec::new()
        };

        Ok(MatchPattern::EnumVariant {
            enum_name,
            variant,
            bindings,
        })
    }

    fn parse_enum_def(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'enum'
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;

        let mut variants = Vec::new();
        while !self.at(&TokenKind::RBrace) && !self.at(&TokenKind::Eof) {
            let (vname, _) = self.expect_ident()?;
            // Optional payload fields: Variant(name: type, name: type, ...)
            let fields = if *self.peek() == TokenKind::LParen {
                self.advance();
                let mut flds = Vec::new();
                if !self.at(&TokenKind::RParen) {
                    loop {
                        let (fname, _) = self.expect_ident()?;
                        self.expect(&TokenKind::Colon)?;
                        let fty = self.parse_type()?;
                        flds.push(EnumVariantField { name: fname, ty: fty });
                        if *self.peek() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&TokenKind::RParen)?;
                flds
            } else {
                Vec::new()
            };
            variants.push(EnumVariant { name: vname, fields });
            if *self.peek() == TokenKind::Comma {
                self.advance();
            }
        }
        self.expect(&TokenKind::RBrace)?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(
            Stmt::EnumDef(EnumDef { name, variants }),
            span,
        ))
    }

    fn parse_import(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'import'

        if *self.peek() == TokenKind::LBrace {
            // import { Name1, Name2 } from module
            self.advance(); // consume '{'
            let mut names = Vec::new();
            loop {
                let (name, _) = self.expect_ident()?;
                names.push(name);
                if *self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::RBrace)?;
            self.expect(&TokenKind::From)?;
            let (module, _) = self.expect_ident()?;
            let end = self.tokens[self.pos - 1].span.end;
            let span = Span::new(start.start, end);
            Ok(self.ids.spanned(Stmt::Import { module, names }, span))
        } else {
            // import module
            let (module, _) = self.expect_ident()?;
            let end = self.tokens[self.pos - 1].span.end;
            let span = Span::new(start.start, end);
            Ok(self.ids.spanned(
                Stmt::Import {
                    module,
                    names: Vec::new(),
                },
                span,
            ))
        }
    }

    fn parse_block(&mut self) -> Result<Block, PdcError> {
        self.expect(&TokenKind::LBrace)?;
        let mut stmts = Vec::new();
        while *self.peek() != TokenKind::RBrace && *self.peek() != TokenKind::Eof {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(Block { stmts })
    }

    fn parse_type(&mut self) -> Result<PdcType, PdcError> {
        if let TokenKind::Ident(name) = self.peek().clone() {
            if name == "Array" {
                self.advance(); // consume "Array"
                self.expect(&TokenKind::Lt)?;
                let elem_ty = self.parse_type()?;
                self.expect(&TokenKind::Gt)?;
                return Ok(PdcType::Array(Box::new(elem_ty)));
            }
            let ty = match name.as_str() {
                "f32" => PdcType::F32,
                "f64" => PdcType::F64,
                "i8" => PdcType::I8,
                "i16" => PdcType::I16,
                "i32" => PdcType::I32,
                "i64" => PdcType::I64,
                "u8" => PdcType::U8,
                "u16" => PdcType::U16,
                "u32" => PdcType::U32,
                "u64" => PdcType::U64,
                "bool" => PdcType::Bool,
                "Path" => PdcType::PathHandle,
                // User-defined struct types (starts with uppercase by convention)
                _ => PdcType::Struct(name),
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
                    self.advance();
                    // Tuple index access: expr.0, expr.1
                    if let TokenKind::IntLit(idx) = self.peek().clone() {
                        self.advance();
                        let span = Span::new(
                            expr.span.start,
                            self.tokens[self.pos - 1].span.end,
                        );
                        expr = self.ids.spanned(
                            Expr::TupleIndex {
                                object: Box::new(expr),
                                index: idx as usize,
                            },
                            span,
                        );
                        continue;
                    }
                    let (method, _) = self.expect_ident()?;
                    if *self.peek() == TokenKind::LParen {
                        self.advance();
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
                        // Field access: expr.name (no parens)
                        let span = Span::new(
                            expr.span.start,
                            self.tokens[self.pos - 1].span.end,
                        );
                        expr = self.ids.spanned(
                            Expr::FieldAccess {
                                object: Box::new(expr),
                                field: method,
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
                // Array<type>() constructor
                if name == "Array" && *self.peek() == TokenKind::Lt {
                    self.advance(); // consume '<'
                    let elem_ty = self.parse_type()?;
                    self.expect(&TokenKind::Gt)?;
                    self.expect(&TokenKind::LParen)?;
                    let args = self.parse_call_args()?;
                    let arg_names = vec![None; args.len()];
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    // Encode the element type in the call name: "Array<f64>"
                    let full_name = format!("Array<{elem_ty}>");
                    return Ok(self.ids.spanned(Expr::Call { name: full_name, args, arg_names }, span));
                }
                if *self.peek() == TokenKind::LParen {
                    self.advance(); // consume '('
                    if self.is_named_arg_start() {
                        // Named args: could be struct construct or named function call
                        let named = self.parse_named_args()?;
                        let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                        let arg_names: Vec<Option<String>> = named.iter().map(|(n, _)| Some(n.clone())).collect();
                        let args: Vec<Spanned<Expr>> = named.into_iter().map(|(_, e)| e).collect();
                        // Parser emits Call with arg_names; type checker resolves
                        // as struct construction or named function call
                        Ok(self.ids.spanned(Expr::Call { name, args, arg_names }, span))
                    } else {
                        let args = self.parse_call_args()?;
                        let arg_names = vec![None; args.len()];
                        let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                        Ok(self.ids.spanned(Expr::Call { name, args, arg_names }, span))
                    }
                } else {
                    Ok(self.ids.spanned(Expr::Variable(name), start))
                }
            }
            TokenKind::LParen => {
                self.advance();
                // Empty tuple: ()
                if *self.peek() == TokenKind::RParen {
                    self.advance();
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    return Ok(self.ids.spanned(Expr::TupleConstruct { elements: Vec::new() }, span));
                }
                let first = self.parse_expr()?;
                if *self.peek() == TokenKind::Comma {
                    // Tuple: (expr, expr, ...)
                    let mut elements = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if *self.peek() == TokenKind::RParen {
                            break; // trailing comma
                        }
                        elements.push(self.parse_expr()?);
                    }
                    self.expect(&TokenKind::RParen)?;
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    Ok(self.ids.spanned(Expr::TupleConstruct { elements }, span))
                } else {
                    // Parenthesized expression
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            _ => Err(PdcError::Parse {
                span: start,
                message: format!("unexpected token {:?}", self.peek()),
            }),
        }
    }

    /// Check if the current position looks like `ident:` (named argument).
    fn is_named_arg_start(&self) -> bool {
        if let TokenKind::Ident(_) = self.peek() {
            if self.pos + 1 < self.tokens.len() {
                return self.tokens[self.pos + 1].kind == TokenKind::Colon;
            }
        }
        false
    }

    /// Parse named arguments: `name: expr, name: expr, ...)`
    fn parse_named_args(&mut self) -> Result<Vec<(String, Spanned<Expr>)>, PdcError> {
        let mut fields = Vec::new();
        if *self.peek() != TokenKind::RParen {
            loop {
                let (name, _) = self.expect_ident()?;
                self.expect(&TokenKind::Colon)?;
                let value = self.parse_expr()?;
                fields.push((name, value));
                if *self.peek() == TokenKind::Comma {
                    self.advance();
                    if *self.peek() == TokenKind::RParen {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(fields)
    }

    fn parse_call_args(&mut self) -> Result<Vec<Spanned<Expr>>, PdcError> {
        let mut args = Vec::new();
        if *self.peek() != TokenKind::RParen {
            args.push(self.parse_expr()?);
            while *self.peek() == TokenKind::Comma {
                self.advance();
                if *self.peek() == TokenKind::RParen {
                    break;
                }
                args.push(self.parse_expr()?);
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok(args)
    }
}
