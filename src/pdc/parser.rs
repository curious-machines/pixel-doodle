use super::ast::*;
use super::error::PdcError;
use super::span::{IdAlloc, Span, Spanned};
use super::token::{NumericSuffix, Token, TokenKind};

fn suffix_to_type(suffix: Option<NumericSuffix>) -> Option<PdcType> {
    suffix.map(|s| match s {
        NumericSuffix::I8 => PdcType::I8,
        NumericSuffix::I16 => PdcType::I16,
        NumericSuffix::I32 => PdcType::I32,
        NumericSuffix::I64 => PdcType::I64,
        NumericSuffix::U8 => PdcType::U8,
        NumericSuffix::U16 => PdcType::U16,
        NumericSuffix::U32 => PdcType::U32,
        NumericSuffix::U64 => PdcType::U64,
        NumericSuffix::F32 => PdcType::F32,
        NumericSuffix::F64 => PdcType::F64,
    })
}

pub fn parse(tokens: Vec<Token>, ids: &mut IdAlloc) -> Result<Vec<Spanned<Stmt>>, PdcError> {
    let mut parser = Parser::new(tokens, ids);
    parser.parse_stmts()
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    ids: &'a mut IdAlloc,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, ids: &'a mut IdAlloc) -> Self {
        Self {
            tokens,
            pos: 0,
            ids,
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

    fn parse_stmts(&mut self) -> Result<Vec<Spanned<Stmt>>, PdcError> {
        let mut stmts = Vec::new();
        while *self.peek() != TokenKind::Eof {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    // ── Statement parsing ──

    fn parse_stmt(&mut self) -> Result<Spanned<Stmt>, PdcError> {
        let start = self.span();

        // Handle `pub` prefix
        if *self.peek() == TokenKind::Pub {
            self.advance(); // consume 'pub'
            let vis = Visibility::Public;
            return match self.peek().clone() {
                TokenKind::Const => self.parse_const_decl(start, vis),
                TokenKind::Var => self.parse_var_decl(start, vis),
                TokenKind::Fn => self.parse_fn_def(start, vis),
                TokenKind::Struct => self.parse_struct_def(start, vis),
                TokenKind::Enum => self.parse_enum_def(start, vis),
                TokenKind::Type => self.parse_type_alias(start, vis),
                TokenKind::Operator => self.parse_operator_def(start, vis),
                _ => Err(PdcError::Parse {
                    span: self.span(),
                    message: format!("'pub' can only precede fn, const, var, struct, enum, type, or operator; got {:?}", self.peek()),
                }),
            };
        }

        let vis = Visibility::Private;
        match self.peek().clone() {
            TokenKind::Builtin => self.parse_builtin_decl(start),
            TokenKind::Const => self.parse_const_decl(start, vis),
            TokenKind::Var => self.parse_var_decl(start, vis),
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
            TokenKind::Struct => self.parse_struct_def(start, vis),
            TokenKind::Enum => self.parse_enum_def(start, vis),
            TokenKind::Type => self.parse_type_alias(start, vis),
            TokenKind::Fn => self.parse_fn_def(start, vis),
            TokenKind::Operator => self.parse_operator_def(start, vis),
            TokenKind::Test => self.parse_test_def(start),
            _ => {
                // Expression statement or assignment
                let expr = self.parse_expr()?;

                // Simple assignment or index assignment
                if *self.peek() == TokenKind::Eq {
                    if let Expr::Variable(name) = &expr.node {
                        let name = name.clone();
                        self.advance();
                        let value = self.parse_expr()?;
                        let span = Span::new(start.start, value.span.end);
                        return Ok(self.ids.spanned(Stmt::Assign { name, value }, span));
                    }
                    if let Expr::Index { object, index } = expr.node {
                        self.advance(); // consume '='
                        let value = self.parse_expr()?;
                        let span = Span::new(start.start, value.span.end);
                        return Ok(self.ids.spanned(
                            Stmt::IndexAssign {
                                object: *object,
                                index: *index,
                                value,
                            },
                            span,
                        ));
                    }
                    if let Expr::FieldAccess { object, field } = expr.node {
                        self.advance(); // consume '='
                        let value = self.parse_expr()?;
                        let span = Span::new(start.start, value.span.end);
                        return Ok(self.ids.spanned(
                            Stmt::FieldAssign {
                                object: *object,
                                field,
                                value,
                            },
                            span,
                        ));
                    }
                }

                // Compound assignment: name += expr → name = name + expr
                let compound_op = match self.peek() {
                    TokenKind::PlusEq => Some(BinOp::Add),
                    TokenKind::MinusEq => Some(BinOp::Sub),
                    TokenKind::StarEq => Some(BinOp::Mul),
                    TokenKind::StarStarEq => Some(BinOp::Pow),
                    TokenKind::SlashEq => Some(BinOp::Div),
                    TokenKind::PercentEq => Some(BinOp::Mod),
                    TokenKind::AmpEq => Some(BinOp::BitAnd),
                    TokenKind::PipeEq => Some(BinOp::BitOr),
                    TokenKind::CaretEq => Some(BinOp::BitXor),
                    TokenKind::LtLtEq => Some(BinOp::Shl),
                    TokenKind::GtGtEq => Some(BinOp::Shr),
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
        let mutable = if *self.peek() == TokenKind::Var {
            self.advance();
            true
        } else {
            self.expect(&TokenKind::Const)?;
            false
        };
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
        Ok(self.ids.spanned(Stmt::BuiltinDecl { name, ty, mutable }, span))
    }

    fn parse_const_decl(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
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
        Ok(self.ids.spanned(Stmt::ConstDecl { vis, name, ty, value }, span))
    }

    fn parse_var_decl(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
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
        Ok(self.ids.spanned(Stmt::VarDecl { vis, name, ty, value }, span))
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
        // Optional const/var qualifier (default: const)
        let mutable = if *self.peek() == TokenKind::Var {
            self.advance();
            true
        } else if *self.peek() == TokenKind::Const {
            self.advance();
            false
        } else {
            false // default to const
        };

        // Check for tuple destructuring: for (a, b) in ...
        let (var_name, destructure_names) = if *self.peek() == TokenKind::LParen {
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
            (String::new(), names)
        } else {
            let (name, _) = self.expect_ident()?;
            (name, Vec::new())
        };

        self.expect(&TokenKind::In)?;
        let expr = self.parse_expr()?;

        if *self.peek() == TokenKind::DotDot || *self.peek() == TokenKind::DotDotEq {
            // Range loop: for i in start..end { } or start..=end { }
            let inclusive = *self.peek() == TokenKind::DotDotEq;
            self.advance();
            let range_end = self.parse_expr()?;
            let body = self.parse_block()?;
            let end = self.tokens[self.pos - 1].span.end;
            let span = Span::new(start.start, end);
            Ok(self.ids.spanned(
                Stmt::For {
                    var_name,
                    mutable,
                    start: expr,
                    end: range_end,
                    inclusive,
                    body,
                },
                span,
            ))
        } else {
            // ForEach: for x in collection { }
            let body = self.parse_block()?;
            let end = self.tokens[self.pos - 1].span.end;
            let span = Span::new(start.start, end);
            Ok(self.ids.spanned(
                Stmt::ForEach {
                    var_name,
                    destructure_names,
                    mutable,
                    collection: expr,
                    body,
                },
                span,
            ))
        }
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

    fn parse_type_alias(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'type'
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Eq)?;
        let ty = self.parse_type()?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(Stmt::TypeAlias { vis, name, ty }, span))
    }

    fn parse_fn_def(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'fn'
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;

        // Parse parameters
        let mut params = Vec::new();
        let mut seen_default = false;
        if !self.at(&TokenKind::RParen) {
            loop {
                let (pname, pspan) = self.expect_ident()?;
                self.expect(&TokenKind::Colon)?;
                let pty = self.parse_type()?;
                let default = if self.at(&TokenKind::Eq) {
                    self.advance(); // consume '='
                    seen_default = true;
                    Some(self.parse_expr()?)
                } else {
                    if seen_default {
                        return Err(PdcError::Parse {
                            span: pspan,
                            message: format!("parameter '{pname}' must have a default value (required parameters cannot follow defaulted ones)"),
                        });
                    }
                    None
                };
                params.push(Param { name: pname, ty: pty, default });
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
                vis,
                name,
                params,
                return_type,
                body,
            }),
            span,
        ))
    }

    fn parse_operator_def(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'operator'

        let op_token = self.peek().clone();
        let base_name = token_to_op_name(&op_token)
            .ok_or_else(|| PdcError::Parse {
                span: self.span(),
                message: format!("expected an operator after 'operator' keyword, got {:?}", op_token),
            })?;
        self.advance(); // consume the operator token

        self.expect(&TokenKind::LParen)?;

        let mut params = Vec::new();
        if !self.at(&TokenKind::RParen) {
            loop {
                let (pname, _) = self.expect_ident()?;
                self.expect(&TokenKind::Colon)?;
                let pty = self.parse_type()?;
                params.push(Param { name: pname, ty: pty, default: None });
                if *self.peek() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;

        // Validate arity and determine final name
        let is_unary_only = matches!(op_token, TokenKind::Bang | TokenKind::Tilde);
        let is_minus = matches!(op_token, TokenKind::Minus);

        let name = if is_unary_only {
            if params.len() != 1 {
                return Err(PdcError::Parse {
                    span: start,
                    message: format!("unary operator requires exactly 1 parameter, got {}", params.len()),
                });
            }
            base_name.to_string()
        } else if is_minus && params.len() == 1 {
            "__op_neg__".to_string()
        } else if params.len() == 2 {
            base_name.to_string()
        } else {
            return Err(PdcError::Parse {
                span: start,
                message: format!(
                    "binary operator requires exactly 2 parameters{}, got {}",
                    if is_minus { " (or 1 for unary -)" } else { "" },
                    params.len(),
                ),
            });
        };

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
                vis,
                name,
                params,
                return_type,
                body,
            }),
            span,
        ))
    }

    fn parse_test_def(&mut self, start: Span) -> Result<Spanned<Stmt>, PdcError> {
        self.advance(); // consume 'test'

        // Expect a string literal for the test name
        let name = match self.peek().clone() {
            TokenKind::StringLit(s) => {
                self.advance();
                s
            }
            _ => {
                return Err(PdcError::Parse {
                    span: self.span(),
                    message: format!("expected string literal for test name, got {:?}", self.peek()),
                });
            }
        };

        let body = self.parse_block()?;
        let end = self.tokens[self.pos - 1].span.end;
        let span = Span::new(start.start, end);
        Ok(self.ids.spanned(Stmt::TestDef { name, body }, span))
    }

    fn parse_struct_def(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
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
            Stmt::StructDef(StructDef { vis, name, fields }),
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

    fn parse_enum_def(&mut self, start: Span, vis: Visibility) -> Result<Spanned<Stmt>, PdcError> {
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
            Stmt::EnumDef(EnumDef { vis, name, variants }),
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
            let module = self.expect_module_name()?;
            let end = self.tokens[self.pos - 1].span.end;
            let span = Span::new(start.start, end);
            Ok(self.ids.spanned(Stmt::Import { module, names }, span))
        } else {
            // import module or import "path/to/module"
            let module = self.expect_module_name()?;
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

    /// Parse a module name: either an identifier (`math`) or a string literal (`"./my_shapes"`).
    fn expect_module_name(&mut self) -> Result<String, PdcError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                Ok(name)
            }
            TokenKind::StringLit(path) => {
                self.advance();
                Ok(path)
            }
            _ => Err(PdcError::Parse {
                span: self.span(),
                message: format!("expected module name (identifier or string), got {:?}", self.peek()),
            }),
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
        // [T] shorthand for Array<T>
        if *self.peek() == TokenKind::LBracket {
            self.advance(); // consume '['
            let elem_ty = self.parse_type()?;
            self.expect(&TokenKind::RBracket)?;
            return Ok(PdcType::Array(Box::new(elem_ty)));
        }

        if let TokenKind::Ident(name) = self.peek().clone() {
            if name == "Array" {
                self.advance(); // consume "Array"
                self.expect(&TokenKind::Lt)?;
                let elem_ty = self.parse_type()?;
                self.expect(&TokenKind::Gt)?;
                return Ok(PdcType::Array(Box::new(elem_ty)));
            }
            if name == "Buffer" {
                self.advance();
                self.expect(&TokenKind::Lt)?;
                let elem_ty = self.parse_type()?;
                self.expect(&TokenKind::Gt)?;
                return Ok(PdcType::BufferHandle(Box::new(elem_ty)));
            }
            if name == "slice" {
                self.advance(); // consume "slice"
                self.expect(&TokenKind::Lt)?;
                let elem_ty = self.parse_type()?;
                self.expect(&TokenKind::Gt)?;
                return Ok(PdcType::Slice(Box::new(elem_ty)));
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
                "vec2f32" => PdcType::Vec2F32,
                "vec3f32" => PdcType::Vec3F32,
                "vec4f32" => PdcType::Vec4F32,
                "string" => PdcType::Str,
                "Path" => PdcType::PathHandle,
                "Kernel" => PdcType::KernelHandle,
                "Texture" => PdcType::TextureHandle,
                "Scene" => PdcType::SceneHandle,
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
        let expr = self.parse_or()?;
        // Ternary: expr ? then : else (right-associative)
        if *self.peek() == TokenKind::Question {
            self.advance();
            let then_expr = self.parse_expr()?;
            self.expect(&TokenKind::Colon)?;
            let else_expr = self.parse_expr()?;
            let span = Span::new(expr.span.start, else_expr.span.end);
            Ok(self.ids.spanned(
                Expr::Ternary {
                    condition: Box::new(expr),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                },
                span,
            ))
        } else {
            Ok(expr)
        }
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
        let mut left = self.parse_bit_or()?;
        while *self.peek() == TokenKind::AmpAmp {
            self.advance();
            let right = self.parse_bit_or()?;
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

    fn parse_bit_or(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_bit_xor()?;
        while *self.peek() == TokenKind::Pipe {
            self.advance();
            let right = self.parse_bit_xor()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op: BinOp::BitOr,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_bit_xor(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_bit_and()?;
        while *self.peek() == TokenKind::Caret {
            self.advance();
            let right = self.parse_bit_and()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op: BinOp::BitXor,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_bit_and(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_comparison()?;
        while *self.peek() == TokenKind::Amp {
            self.advance();
            let right = self.parse_comparison()?;
            let span = Span::new(left.span.start, right.span.end);
            left = self.ids.spanned(
                Expr::BinaryOp {
                    op: BinOp::BitAnd,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_shift()?;
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
            let right = self.parse_shift()?;
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

    fn parse_shift(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let mut left = self.parse_addition()?;
        loop {
            let op = match self.peek() {
                TokenKind::LtLt => BinOp::Shl,
                TokenKind::GtGt => BinOp::Shr,
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
        let mut left = self.parse_exponent()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_exponent()?;
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

    /// Exponentiation is right-associative: `a ** b ** c` = `a ** (b ** c)`
    fn parse_exponent(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let left = self.parse_unary()?;
        if *self.peek() == TokenKind::StarStar {
            self.advance();
            let right = self.parse_exponent()?; // right-recursive for right-associativity
            let span = Span::new(left.span.start, right.span.end);
            Ok(self.ids.spanned(
                Expr::BinaryOp {
                    op: BinOp::Pow,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            ))
        } else {
            Ok(left)
        }
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
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_unary()?;
                let span = Span::new(start.start, operand.span.end);
                Ok(self.ids.spanned(
                    Expr::UnaryOp {
                        op: UnaryOp::BitNot,
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
                    if let TokenKind::IntLit(idx, _) = self.peek().clone() {
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
                        let (args, _arg_names) = self.parse_call_args()?;
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
                TokenKind::LBracket => {
                    self.advance(); // consume '['
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    let span = Span::new(
                        expr.span.start,
                        self.tokens[self.pos - 1].span.end,
                    );
                    expr = self.ids.spanned(
                        Expr::Index {
                            object: Box::new(expr),
                            index: Box::new(index),
                        },
                        span,
                    );
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Spanned<Expr>, PdcError> {
        let start = self.span();
        match self.peek().clone() {
            TokenKind::IntLit(val, suffix) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::Int(val, suffix_to_type(suffix))), start))
            }
            TokenKind::FloatLit(val, suffix) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::Float(val, suffix_to_type(suffix))), start))
            }
            TokenKind::BoolLit(val) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::Bool(val)), start))
            }
            TokenKind::StringLit(val) => {
                self.advance();
                Ok(self.ids.spanned(Expr::Literal(Literal::String(val)), start))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Array<type>() constructor
                if name == "Array" && *self.peek() == TokenKind::Lt {
                    self.advance(); // consume '<'
                    let elem_ty = self.parse_type()?;
                    self.expect(&TokenKind::Gt)?;
                    self.expect(&TokenKind::LParen)?;
                    let (args, arg_names) = self.parse_call_args()?;
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    // Encode the element type in the call name: "Array<f64>"
                    let full_name = format!("Array<{elem_ty}>");
                    return Ok(self.ids.spanned(Expr::Call { name: full_name, args, arg_names }, span));
                }
                // Buffer<type>() constructor
                if name == "Buffer" && *self.peek() == TokenKind::Lt {
                    self.advance(); // consume '<'
                    let elem_ty = self.parse_type()?;
                    self.expect(&TokenKind::Gt)?;
                    self.expect(&TokenKind::LParen)?;
                    let (args, arg_names) = self.parse_call_args()?;
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    let full_name = format!("Buffer<{elem_ty}>");
                    return Ok(self.ids.spanned(Expr::Call { name: full_name, args, arg_names }, span));
                }
                if *self.peek() == TokenKind::LParen {
                    self.advance(); // consume '('
                    let (args, arg_names) = self.parse_call_args()?;
                    let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                    Ok(self.ids.spanned(Expr::Call { name, args, arg_names }, span))
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
            TokenKind::Dot => {
                self.advance(); // consume '.'
                let (variant, _) = self.expect_ident()?;
                let span = Span::new(start.start, self.tokens[self.pos - 1].span.end);
                Ok(self.ids.spanned(Expr::DotShorthand(variant), span))
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

    /// Parse call arguments supporting mixed positional + named args.
    /// Positional args must come first; once a named arg appears, all
    /// subsequent args must also be named.
    fn parse_call_args(&mut self) -> Result<(Vec<Spanned<Expr>>, Vec<Option<String>>), PdcError> {
        let mut args = Vec::new();
        let mut arg_names: Vec<Option<String>> = Vec::new();
        let mut seen_named = false;

        if *self.peek() != TokenKind::RParen {
            loop {
                if self.is_named_arg_start() {
                    seen_named = true;
                    let (name, _) = self.expect_ident()?;
                    self.expect(&TokenKind::Colon)?;
                    let value = self.parse_expr()?;
                    arg_names.push(Some(name));
                    args.push(value);
                } else {
                    if seen_named {
                        return Err(PdcError::Parse {
                            span: self.tokens[self.pos].span,
                            message: "positional arguments must come before named arguments".into(),
                        });
                    }
                    let value = self.parse_expr()?;
                    arg_names.push(None);
                    args.push(value);
                }
                if *self.peek() == TokenKind::Comma {
                    self.advance();
                    if *self.peek() == TokenKind::RParen {
                        break; // trailing comma
                    }
                } else {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;
        Ok((args, arg_names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::lexer;
    use super::super::span::IdAlloc;

    /// Parse source and return the AST statements.
    fn parse_ok(source: &str) -> Vec<Spanned<Stmt>> {
        let tokens = lexer::lex(source).expect("lex failed");
        let mut ids = IdAlloc::new();
        parse(tokens, &mut ids).expect("parse failed")
    }

    /// Parse source and expect an error containing the given substring.
    fn parse_err(source: &str, expected_substr: &str) {
        let tokens = match lexer::lex(source) {
            Ok(t) => t,
            Err(e) => {
                // Lex error is also acceptable for parse_err
                let msg = e.to_string();
                assert!(msg.contains(expected_substr),
                    "lex error '{}' does not contain '{}'", msg, expected_substr);
                return;
            }
        };
        let mut ids = IdAlloc::new();
        match parse(tokens, &mut ids) {
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains(expected_substr),
                    "error '{}' does not contain '{}'", msg, expected_substr);
            }
            Ok(_) => panic!("expected error containing '{}'", expected_substr),
        }
    }

    /// Parse a single statement.
    fn parse_one(source: &str) -> Stmt {
        let stmts = parse_ok(source);
        assert_eq!(stmts.len(), 1, "expected 1 statement, got {}", stmts.len());
        stmts.into_iter().next().unwrap().node
    }

    // ---- Variable and constant declarations ----

    #[test]
    fn const_decl_with_type() {
        match parse_one("const x: f64 = 3.14") {
            Stmt::ConstDecl { name, ty, .. } => {
                assert_eq!(name, "x");
                assert_eq!(ty, Some(PdcType::F64));
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn const_decl_without_type() {
        match parse_one("const x = 42") {
            Stmt::ConstDecl { name, ty, .. } => {
                assert_eq!(name, "x");
                assert_eq!(ty, None);
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn var_decl() {
        match parse_one("var y: i32 = 0") {
            Stmt::VarDecl { name, ty, .. } => {
                assert_eq!(name, "y");
                assert_eq!(ty, Some(PdcType::I32));
            }
            other => panic!("expected VarDecl, got {:?}", other),
        }
    }

    #[test]
    fn pub_const_decl() {
        match parse_one("pub const PI = 3.14159") {
            Stmt::ConstDecl { vis, name, .. } => {
                assert_eq!(vis, Visibility::Public);
                assert_eq!(name, "PI");
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn builtin_decl() {
        match parse_one("builtin const width: f32") {
            Stmt::BuiltinDecl { name, ty, mutable } => {
                assert_eq!(name, "width");
                assert_eq!(ty, PdcType::F32);
                assert!(!mutable);
            }
            other => panic!("expected BuiltinDecl, got {:?}", other),
        }
    }

    #[test]
    fn builtin_var_decl() {
        match parse_one("builtin var paused: bool") {
            Stmt::BuiltinDecl { name, ty, mutable } => {
                assert_eq!(name, "paused");
                assert_eq!(ty, PdcType::Bool);
                assert!(mutable);
            }
            other => panic!("expected BuiltinDecl, got {:?}", other),
        }
    }

    // ---- Assignments ----

    #[test]
    fn simple_assign() {
        match parse_one("x = 5") {
            Stmt::Assign { name, .. } => assert_eq!(name, "x"),
            other => panic!("expected Assign, got {:?}", other),
        }
    }

    #[test]
    fn index_assign() {
        let stmts = parse_ok("var arr: Array<i32> = Array()\narr[0] = 42");
        assert_eq!(stmts.len(), 2);
        match &stmts[1].node {
            Stmt::IndexAssign { .. } => {}
            other => panic!("expected IndexAssign, got {:?}", other),
        }
    }

    #[test]
    fn compound_assign_plus() {
        // x += 1 desugars to x = x + 1
        let stmts = parse_ok("var x = 0\nx += 1");
        match &stmts[1].node {
            Stmt::Assign { name, value } => {
                assert_eq!(name, "x");
                match &value.node {
                    Expr::BinaryOp { op, .. } => assert_eq!(*op, BinOp::Add),
                    other => panic!("expected BinaryOp, got {:?}", other),
                }
            }
            other => panic!("expected Assign, got {:?}", other),
        }
    }

    // ---- Function definitions ----

    #[test]
    fn fn_def_no_params() {
        match parse_one("fn foo() { return 42 }") {
            Stmt::FnDef(fndef) => {
                assert_eq!(fndef.name, "foo");
                assert_eq!(fndef.params.len(), 0);
                assert_eq!(fndef.return_type, PdcType::Void);
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    #[test]
    fn fn_def_with_params_and_return() {
        match parse_one("fn add(a: f64, b: f64) -> f64 { return a + b }") {
            Stmt::FnDef(fndef) => {
                assert_eq!(fndef.name, "add");
                assert_eq!(fndef.params.len(), 2);
                assert_eq!(fndef.params[0].name, "a");
                assert_eq!(fndef.params[0].ty, PdcType::F64);
                assert_eq!(fndef.params[1].name, "b");
                assert_eq!(fndef.params[1].ty, PdcType::F64);
                assert_eq!(fndef.return_type, PdcType::F64);
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    #[test]
    fn pub_fn_def() {
        match parse_one("pub fn greet() { }") {
            Stmt::FnDef(fndef) => {
                assert_eq!(fndef.vis, Visibility::Public);
                assert_eq!(fndef.name, "greet");
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    // ---- Control flow ----

    #[test]
    fn if_statement() {
        match parse_one("if x > 0 { return 1 }") {
            Stmt::If { elsif_clauses, else_body, .. } => {
                assert!(elsif_clauses.is_empty());
                assert!(else_body.is_none());
            }
            other => panic!("expected If, got {:?}", other),
        }
    }

    #[test]
    fn if_else() {
        match parse_one("if x > 0 { return 1 } else { return 0 }") {
            Stmt::If { else_body, .. } => {
                assert!(else_body.is_some());
            }
            other => panic!("expected If, got {:?}", other),
        }
    }

    #[test]
    fn if_elsif_else() {
        match parse_one("if x > 0 { return 1 } elsif x < 0 { return -1 } else { return 0 }") {
            Stmt::If { elsif_clauses, else_body, .. } => {
                assert_eq!(elsif_clauses.len(), 1);
                assert!(else_body.is_some());
            }
            other => panic!("expected If, got {:?}", other),
        }
    }

    #[test]
    fn while_loop() {
        match parse_one("while x > 0 { x = x - 1 }") {
            Stmt::While { .. } => {}
            other => panic!("expected While, got {:?}", other),
        }
    }

    #[test]
    fn for_range_exclusive() {
        match parse_one("for i in 0..10 { x = i }") {
            Stmt::For { var_name, inclusive, .. } => {
                assert_eq!(var_name, "i");
                assert!(!inclusive);
            }
            other => panic!("expected For, got {:?}", other),
        }
    }

    #[test]
    fn for_range_inclusive() {
        match parse_one("for i in 0..=10 { x = i }") {
            Stmt::For { var_name, inclusive, .. } => {
                assert_eq!(var_name, "i");
                assert!(inclusive);
            }
            other => panic!("expected For, got {:?}", other),
        }
    }

    #[test]
    fn for_each_loop() {
        let stmts = parse_ok("var arr = Array()\nfor x in arr { }");
        match &stmts[1].node {
            Stmt::ForEach { var_name, .. } => assert_eq!(var_name, "x"),
            other => panic!("expected ForEach, got {:?}", other),
        }
    }

    #[test]
    fn loop_statement() {
        match parse_one("loop { break }") {
            Stmt::Loop { body } => {
                assert_eq!(body.stmts.len(), 1);
            }
            other => panic!("expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn break_statement() {
        match parse_one("break") {
            Stmt::Break => {}
            other => panic!("expected Break, got {:?}", other),
        }
    }

    #[test]
    fn continue_statement() {
        match parse_one("continue") {
            Stmt::Continue => {}
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    #[test]
    fn return_with_value() {
        match parse_one("return 42") {
            Stmt::Return(Some(_)) => {}
            other => panic!("expected Return with value, got {:?}", other),
        }
    }

    #[test]
    fn return_no_value() {
        match parse_one("return") {
            Stmt::Return(None) => {}
            other => panic!("expected Return without value, got {:?}", other),
        }
    }

    // ---- Struct and enum definitions ----

    #[test]
    fn struct_def() {
        match parse_one("struct Point { x: f64, y: f64 }") {
            Stmt::StructDef(sd) => {
                assert_eq!(sd.name, "Point");
                assert_eq!(sd.fields.len(), 2);
                assert_eq!(sd.fields[0].name, "x");
                assert_eq!(sd.fields[1].name, "y");
            }
            other => panic!("expected StructDef, got {:?}", other),
        }
    }

    #[test]
    fn enum_def_simple() {
        match parse_one("enum Color { Red, Green, Blue }") {
            Stmt::EnumDef(ed) => {
                assert_eq!(ed.name, "Color");
                assert_eq!(ed.variants.len(), 3);
                assert_eq!(ed.variants[0].name, "Red");
                assert_eq!(ed.variants[1].name, "Green");
                assert_eq!(ed.variants[2].name, "Blue");
            }
            other => panic!("expected EnumDef, got {:?}", other),
        }
    }

    #[test]
    fn type_alias() {
        match parse_one("type Real = f64") {
            Stmt::TypeAlias { name, ty, .. } => {
                assert_eq!(name, "Real");
                assert_eq!(ty, PdcType::F64);
            }
            other => panic!("expected TypeAlias, got {:?}", other),
        }
    }

    // ---- Import statements ----

    #[test]
    fn import_module() {
        match parse_one("import math") {
            Stmt::Import { module, names } => {
                assert_eq!(module, "math");
                assert!(names.is_empty());
            }
            other => panic!("expected Import, got {:?}", other),
        }
    }

    #[test]
    fn import_named() {
        match parse_one("import { sin, cos } from math") {
            Stmt::Import { module, names } => {
                assert_eq!(module, "math");
                assert_eq!(names, vec!["sin", "cos"]);
            }
            other => panic!("expected Import, got {:?}", other),
        }
    }

    // ---- Expressions ----

    #[test]
    fn binary_op_precedence() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        match parse_one("const x = 1 + 2 * 3") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::BinaryOp { op, right, .. } => {
                        assert_eq!(*op, BinOp::Add);
                        // right side should be 2 * 3
                        match &right.node {
                            Expr::BinaryOp { op, .. } => assert_eq!(*op, BinOp::Mul),
                            other => panic!("expected BinaryOp, got {:?}", other),
                        }
                    }
                    other => panic!("expected BinaryOp, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn unary_negation() {
        match parse_one("const x = -5") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::UnaryOp { op, .. } => assert_eq!(*op, UnaryOp::Neg),
                    other => panic!("expected UnaryOp, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn logical_not() {
        match parse_one("const x = !true") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::UnaryOp { op, .. } => assert_eq!(*op, UnaryOp::Not),
                    other => panic!("expected UnaryOp, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn bitwise_not() {
        match parse_one("const x = ~0xFF") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::UnaryOp { op, .. } => assert_eq!(*op, UnaryOp::BitNot),
                    other => panic!("expected UnaryOp, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn function_call_expr() {
        match parse_one("foo(1, 2, 3)") {
            Stmt::ExprStmt(expr) => {
                match &expr.node {
                    Expr::Call { name, args, .. } => {
                        assert_eq!(name, "foo");
                        assert_eq!(args.len(), 3);
                    }
                    other => panic!("expected Call, got {:?}", other),
                }
            }
            other => panic!("expected ExprStmt, got {:?}", other),
        }
    }

    #[test]
    fn method_call_expr() {
        match parse_one("const x = obj.method(1)") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::MethodCall { method, args, .. } => {
                        assert_eq!(method, "method");
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("expected MethodCall, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn field_access_expr() {
        match parse_one("const x = point.x") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::FieldAccess { field, .. } => {
                        assert_eq!(field, "x");
                    }
                    other => panic!("expected FieldAccess, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn index_expr() {
        match parse_one("const x = arr[0]") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::Index { .. } => {}
                    other => panic!("expected Index, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn ternary_expr() {
        match parse_one("const x = a > b ? a : b") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::Ternary { .. } => {}
                    other => panic!("expected Ternary, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn comparison_operators() {
        let ops = vec![
            ("a == b", BinOp::Eq),
            ("a != b", BinOp::NotEq),
            ("a < b", BinOp::Lt),
            ("a <= b", BinOp::LtEq),
            ("a > b", BinOp::Gt),
            ("a >= b", BinOp::GtEq),
        ];
        for (src, expected_op) in ops {
            let full = format!("const x = {}", src);
            match parse_one(&full) {
                Stmt::ConstDecl { value, .. } => match &value.node {
                    Expr::BinaryOp { op, .. } => assert_eq!(*op, expected_op, "for: {}", src),
                    other => panic!("expected BinaryOp for '{}', got {:?}", src, other),
                },
                other => panic!("expected ConstDecl for '{}', got {:?}", src, other),
            }
        }
    }

    #[test]
    fn logical_operators() {
        let ops = vec![
            ("a && b", BinOp::And),
            ("a || b", BinOp::Or),
        ];
        for (src, expected_op) in ops {
            let full = format!("const x = {}", src);
            match parse_one(&full) {
                Stmt::ConstDecl { value, .. } => match &value.node {
                    Expr::BinaryOp { op, .. } => assert_eq!(*op, expected_op, "for: {}", src),
                    other => panic!("expected BinaryOp for '{}', got {:?}", src, other),
                },
                other => panic!("expected ConstDecl for '{}', got {:?}", src, other),
            }
        }
    }

    #[test]
    fn bitwise_operators() {
        let ops = vec![
            ("a & b", BinOp::BitAnd),
            ("a | b", BinOp::BitOr),
            ("a ^ b", BinOp::BitXor),
            ("a << b", BinOp::Shl),
            ("a >> b", BinOp::Shr),
        ];
        for (src, expected_op) in ops {
            let full = format!("const x = {}", src);
            match parse_one(&full) {
                Stmt::ConstDecl { value, .. } => match &value.node {
                    Expr::BinaryOp { op, .. } => assert_eq!(*op, expected_op, "for: {}", src),
                    other => panic!("expected BinaryOp for '{}', got {:?}", src, other),
                },
                other => panic!("expected ConstDecl for '{}', got {:?}", src, other),
            }
        }
    }

    #[test]
    fn power_operator() {
        match parse_one("const x = 2 ** 10") {
            Stmt::ConstDecl { value, .. } => match &value.node {
                Expr::BinaryOp { op, .. } => assert_eq!(*op, BinOp::Pow),
                other => panic!("expected BinaryOp, got {:?}", other),
            },
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    // ---- Tuple destructuring ----

    #[test]
    fn tuple_destructure_const() {
        match parse_one("const (a, b) = pair") {
            Stmt::TupleDestructure { names, is_const, .. } => {
                assert_eq!(names, vec!["a", "b"]);
                assert!(is_const);
            }
            other => panic!("expected TupleDestructure, got {:?}", other),
        }
    }

    #[test]
    fn tuple_destructure_var() {
        match parse_one("var (x, y, z) = triple") {
            Stmt::TupleDestructure { names, is_const, .. } => {
                assert_eq!(names, vec!["x", "y", "z"]);
                assert!(!is_const);
            }
            other => panic!("expected TupleDestructure, got {:?}", other),
        }
    }

    // ---- Match statement ----

    #[test]
    fn match_statement() {
        let src = "match color { Color.Red => { x = 1 }, Color.Blue => { x = 2 } }";
        match parse_one(src) {
            Stmt::Match { arms, .. } => {
                assert_eq!(arms.len(), 2);
            }
            other => panic!("expected Match, got {:?}", other),
        }
    }

    // ---- Edge cases ----

    #[test]
    fn empty_block() {
        match parse_one("fn nothing() { }") {
            Stmt::FnDef(fndef) => {
                assert!(fndef.body.stmts.is_empty());
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    #[test]
    fn nested_if() {
        // Should parse without error
        let _ = parse_ok("if a { if b { x = 1 } }");
    }

    #[test]
    fn chained_method_calls() {
        // obj.a().b().c() — should parse as nested method calls
        match parse_one("const x = obj.a().b().c()") {
            Stmt::ConstDecl { value, .. } => {
                match &value.node {
                    Expr::MethodCall { method, .. } => assert_eq!(method, "c"),
                    other => panic!("expected MethodCall, got {:?}", other),
                }
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    #[test]
    fn multiple_statements() {
        let stmts = parse_ok("const a = 1\nconst b = 2\nconst c = 3");
        assert_eq!(stmts.len(), 3);
    }

    // ---- Error cases ----

    #[test]
    fn error_missing_equals_in_const() {
        parse_err("const x 5", "expected");
    }

    #[test]
    fn error_pub_before_invalid() {
        parse_err("pub break", "pub");
    }

    #[test]
    fn error_missing_brace() {
        parse_err("if x > 0 { return 1", "expected");
    }

    // ---- All PDC types parse correctly ----

    #[test]
    fn all_numeric_types() {
        let types = vec![
            ("f32", PdcType::F32),
            ("f64", PdcType::F64),
            ("i8", PdcType::I8),
            ("i16", PdcType::I16),
            ("i32", PdcType::I32),
            ("i64", PdcType::I64),
            ("u8", PdcType::U8),
            ("u16", PdcType::U16),
            ("u32", PdcType::U32),
            ("u64", PdcType::U64),
            ("bool", PdcType::Bool),
        ];
        for (name, expected_ty) in types {
            let src = format!("const x: {} = 0", name);
            match parse_one(&src) {
                Stmt::ConstDecl { ty, .. } => {
                    assert_eq!(ty, Some(expected_ty), "for type: {}", name);
                }
                other => panic!("expected ConstDecl for type {}, got {:?}", name, other),
            }
        }
    }

    #[test]
    fn string_type() {
        match parse_one("const s: string = \"hello\"") {
            Stmt::ConstDecl { ty, .. } => {
                assert_eq!(ty, Some(PdcType::Str));
            }
            other => panic!("expected ConstDecl, got {:?}", other),
        }
    }

    // ---- Array shorthand syntax ----

    #[test]
    fn array_shorthand_type() {
        // [f64] should parse as Array<f64>
        match parse_one("fn foo(points: [f64]) { }") {
            Stmt::FnDef(fndef) => {
                assert_eq!(fndef.params.len(), 1);
                assert_eq!(fndef.params[0].ty, PdcType::Array(Box::new(PdcType::F64)));
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    #[test]
    fn array_shorthand_nested() {
        // [[i32]] should parse as Array<Array<i32>>
        match parse_one("fn foo(matrix: [[i32]]) { }") {
            Stmt::FnDef(fndef) => {
                assert_eq!(fndef.params[0].ty,
                    PdcType::Array(Box::new(PdcType::Array(Box::new(PdcType::I32)))));
            }
            other => panic!("expected FnDef, got {:?}", other),
        }
    }

    #[test]
    fn array_verbose_and_shorthand_equivalent() {
        // Array<f64> and [f64] should produce the same type
        let verbose = parse_one("fn foo(a: Array<f64>) { }");
        let shorthand = parse_one("fn bar(a: [f64]) { }");
        match (verbose, shorthand) {
            (Stmt::FnDef(v), Stmt::FnDef(s)) => {
                assert_eq!(v.params[0].ty, s.params[0].ty);
            }
            _ => panic!("expected FnDef"),
        }
    }
}
