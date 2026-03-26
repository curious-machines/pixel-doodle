use crate::kernel_ir::{ValType, ScalarType};
use super::ast::*;
use super::lexer::{self, Token, Spanned};
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    /// Directory of the file being parsed, for resolving relative `use` paths.
    base_dir: PathBuf,
    /// Set of already-included canonical paths (shared across recursive parses).
    included: Rc<RefCell<HashSet<PathBuf>>>,
}

#[derive(Debug)]
pub struct ParseError {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.message)
    }
}

impl Parser {
    pub fn new(tokens: Vec<Spanned>) -> Self {
        Self {
            tokens,
            pos: 0,
            base_dir: PathBuf::from("."),
            included: Rc::new(RefCell::new(HashSet::new())),
        }
    }

    pub fn new_with_context(
        tokens: Vec<Spanned>,
        base_dir: PathBuf,
        included: Rc<RefCell<HashSet<PathBuf>>>,
    ) -> Self {
        Self { tokens, pos: 0, base_dir, included }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn span(&self) -> Span {
        Span {
            line: self.tokens[self.pos].line,
            col: self.tokens[self.pos].col,
        }
    }

    fn advance(&mut self) -> &Spanned {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<Span, ParseError> {
        let span = self.span();
        if self.peek() == expected {
            self.advance();
            Ok(span)
        } else {
            Err(self.error(format!("expected '{}', got '{}'", expected, self.peek())))
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        let span = self.span();
        if let Token::Ident(name) = self.peek().clone() {
            self.advance();
            Ok((name, span))
        } else {
            Err(self.error(format!("expected identifier, got '{}'", self.peek())))
        }
    }

    fn error(&self, message: String) -> ParseError {
        ParseError {
            line: self.tokens[self.pos].line,
            col: self.tokens[self.pos].col,
            message,
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut fns = Vec::new();
        let mut kernel = None;

        loop {
            match self.peek() {
                Token::Use => {
                    let use_fns = self.parse_use()?;
                    fns.extend(use_fns);
                }
                Token::Fn => fns.push(self.parse_fn_def()?),
                Token::Kernel => {
                    if kernel.is_some() {
                        return Err(self.error("only one kernel allowed per file".into()));
                    }
                    kernel = Some(self.parse_kernel_def()?);
                }
                Token::Eof => break,
                _ => return Err(self.error(format!("expected 'use', 'fn', or 'kernel', got '{}'", self.peek()))),
            }
        }

        let kernel = kernel.ok_or_else(|| self.error("no kernel defined".into()))?;
        Ok(Program { fns, kernel })
    }

    /// Parse `use "path";` and return the function definitions from the included file.
    fn parse_use(&mut self) -> Result<Vec<FnDef>, ParseError> {
        let span = self.span();
        self.advance(); // use

        // Expect string literal
        let path_str = match self.peek().clone() {
            Token::StringLit(s) => { self.advance(); s }
            _ => return Err(self.error(format!("expected string path after 'use', got '{}'", self.peek()))),
        };
        self.expect(&Token::Semi)?;

        // Resolve relative to current file's directory
        let resolved = self.base_dir.join(&path_str);
        let canonical = resolved.canonicalize().map_err(|e| {
            ParseError {
                line: span.line,
                col: span.col,
                message: format!("cannot open '{}': {}", path_str, e),
            }
        })?;

        // Dedup: skip if already included
        if self.included.borrow().contains(&canonical) {
            return Ok(Vec::new());
        }
        self.included.borrow_mut().insert(canonical.clone());

        // Read and lex the included file
        let source = std::fs::read_to_string(&canonical).map_err(|e| {
            ParseError {
                line: span.line,
                col: span.col,
                message: format!("cannot read '{}': {}", path_str, e),
            }
        })?;
        let tokens = lexer::lex(&source).map_err(|e| {
            ParseError {
                line: span.line,
                col: span.col,
                message: format!("in '{}': {}", path_str, e),
            }
        })?;

        // Parse the included file
        let sub_dir = canonical.parent().unwrap().to_path_buf();
        let mut sub = Parser::new_with_context(tokens, sub_dir, self.included.clone());
        let sub_prog = sub.parse_library(&path_str).map_err(|e| {
            ParseError {
                line: span.line,
                col: span.col,
                message: format!("in '{}': {}", path_str, e.message),
            }
        })?;

        Ok(sub_prog)
    }

    /// Parse an included file: allows `use` and `fn` but not `kernel`.
    fn parse_library(&mut self, file_name: &str) -> Result<Vec<FnDef>, ParseError> {
        let mut fns = Vec::new();

        loop {
            match self.peek() {
                Token::Use => {
                    let use_fns = self.parse_use()?;
                    fns.extend(use_fns);
                }
                Token::Fn => fns.push(self.parse_fn_def()?),
                Token::Kernel => {
                    return Err(self.error(format!(
                        "included file '{}' must not contain a kernel", file_name
                    )));
                }
                Token::Eof => break,
                _ => return Err(self.error(format!("expected 'use', 'fn', or end of file, got '{}'", self.peek()))),
            }
        }

        Ok(fns)
    }

    fn parse_type(&mut self) -> Result<ValType, ParseError> {
        match self.peek() {
            Token::TyF64 => { self.advance(); Ok(ValType::F64) }
            Token::TyU32 => { self.advance(); Ok(ValType::U32) }
            Token::TyBool => { self.advance(); Ok(ValType::BOOL) }
            Token::TyVec2 | Token::TyVec3 => {
                let len: u8 = if *self.peek() == Token::TyVec2 { 2 } else { 3 };
                self.advance();
                self.expect(&Token::Lt)?;
                let elem = self.parse_scalar_type()?;
                self.expect(&Token::Gt)?;
                Ok(ValType::Vec { len, elem })
            }
            _ => Err(self.error(format!("expected type, got '{}'", self.peek()))),
        }
    }

    fn parse_scalar_type(&mut self) -> Result<ScalarType, ParseError> {
        match self.peek() {
            Token::TyF64 => { self.advance(); Ok(ScalarType::F64) }
            Token::TyU32 => { self.advance(); Ok(ScalarType::U32) }
            Token::TyBool => { self.advance(); Ok(ScalarType::Bool) }
            _ => Err(self.error(format!("expected scalar type, got '{}'", self.peek()))),
        }
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        if *self.peek() != Token::RParen {
            loop {
                let (name, _) = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let ty = self.parse_type()?;
                params.push(Param { name, ty });
                if *self.peek() != Token::Comma {
                    break;
                }
                self.advance();
            }
        }
        self.expect(&Token::RParen)?;
        Ok(params)
    }

    fn parse_fn_def(&mut self) -> Result<FnDef, ParseError> {
        let span = self.span();
        self.expect(&Token::Fn)?;
        let (name, _) = self.expect_ident()?;
        let params = self.parse_params()?;
        self.expect(&Token::Arrow)?;
        let return_ty = self.parse_type()?;
        self.expect(&Token::LBrace)?;
        let body = self.parse_body()?;
        self.expect(&Token::RBrace)?;
        Ok(FnDef { name, params, return_ty, body, span })
    }

    fn parse_kernel_def(&mut self) -> Result<KernelDef, ParseError> {
        let span = self.span();
        self.expect(&Token::Kernel)?;
        let (name, _) = self.expect_ident()?;
        let params = self.parse_params()?;
        self.expect(&Token::Arrow)?;
        let return_ty = self.parse_type()?;
        let buffers = if *self.peek() == Token::Buffers {
            self.parse_buffers()?
        } else {
            Vec::new()
        };
        self.expect(&Token::LBrace)?;
        let body = self.parse_body()?;
        self.expect(&Token::RBrace)?;
        Ok(KernelDef { name, params, return_ty, buffers, body, span })
    }

    fn parse_buffers(&mut self) -> Result<Vec<BufferParam>, ParseError> {
        self.advance(); // buffers
        self.expect(&Token::LParen)?;
        let mut bufs = Vec::new();
        loop {
            if *self.peek() == Token::RParen {
                break;
            }
            let (name, _) = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let is_output = match self.peek() {
                Token::Read => { self.advance(); false }
                Token::Write => { self.advance(); true }
                _ => return Err(self.error(format!("expected 'read' or 'write', got '{}'", self.peek()))),
            };
            bufs.push(BufferParam { name, is_output });
            if *self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        Ok(bufs)
    }

    fn parse_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let mut stmts = Vec::new();
        loop {
            match self.peek() {
                Token::RBrace | Token::Eof => break,
                _ => stmts.push(self.parse_stmt()?),
            }
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek() {
            Token::Let => self.parse_let(),
            Token::While => self.parse_while(),
            Token::BreakIf => self.parse_break_if(),
            Token::Yield => self.parse_yield(),
            Token::Emit => self.parse_emit(),
            Token::Return => self.parse_return(),
            Token::Ident(name) if name == "buf_store" => self.parse_buf_store(),
            _ => Err(self.error(format!("expected statement, got '{}'", self.peek()))),
        }
    }

    fn parse_buf_store(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        self.advance(); // buf_store
        self.expect(&Token::LParen)?;
        let (buf_name, _) = self.expect_ident()?;
        self.expect(&Token::Comma)?;
        let x = self.parse_expr()?;
        self.expect(&Token::Comma)?;
        let y = self.parse_expr()?;
        self.expect(&Token::Comma)?;
        let val = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::BufStore { buf_name, x, y, val, span })
    }

    fn parse_let(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        self.advance(); // let
        let (name, _) = self.expect_ident()?;
        let ty = if *self.peek() == Token::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&Token::Eq)?;
        let expr = self.parse_expr()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::Let { name, ty, expr, span })
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        // `return expr;` is sugar for `emit expr;` in fn context
        // The lowerer handles this — at the AST level we treat it as emit
        let span = self.span();
        self.advance(); // return
        let expr = self.parse_expr()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::Emit { expr, span })
    }

    fn parse_while(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        self.advance(); // while
        let mut carry = Vec::new();
        // Parse carry variables: `name = expr` or `name: type = expr`
        loop {
            let (name, cspan) = self.expect_ident()?;
            let ty = if *self.peek() == Token::Colon {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(&Token::Eq)?;
            let init = self.parse_expr()?;
            carry.push(CarryDef { name, ty, init, span: cspan });
            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }
        self.expect(&Token::LBrace)?;
        let body = self.parse_body()?;
        self.expect(&Token::RBrace)?;
        Ok(Stmt::While { carry, body, span })
    }

    fn parse_break_if(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        self.advance(); // break_if
        let cond = self.parse_expr()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::BreakIf { cond, span })
    }

    fn parse_yield(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        self.advance(); // yield
        let mut values = Vec::new();
        loop {
            values.push(self.parse_expr()?);
            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }
        self.expect(&Token::Semi)?;
        Ok(Stmt::Yield { values, span })
    }

    fn parse_emit(&mut self) -> Result<Stmt, ParseError> {
        let span = self.span();
        self.advance(); // emit
        let expr = self.parse_expr()?;
        self.expect(&Token::Semi)?;
        Ok(Stmt::Emit { expr, span })
    }

    // ── Expression parsing (Pratt parser) ──

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_precedence(0)
    }

    fn parse_precedence(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;

        loop {
            let (op, bp) = match self.peek() {
                Token::PipePipe => (BinOpKind::Or, (2, 3)),
                Token::AmpAmp => (BinOpKind::And, (4, 5)),
                Token::EqEq => (BinOpKind::Eq, (6, 7)),
                Token::BangEq => (BinOpKind::Ne, (6, 7)),
                Token::Lt => (BinOpKind::Lt, (8, 9)),
                Token::Le => (BinOpKind::Le, (8, 9)),
                Token::Gt => (BinOpKind::Gt, (8, 9)),
                Token::Ge => (BinOpKind::Ge, (8, 9)),
                Token::Pipe => (BinOpKind::BitOr, (10, 11)),
                Token::Caret => (BinOpKind::BitXor, (12, 13)),
                Token::Amp => (BinOpKind::BitAnd, (14, 15)),
                Token::Shl => (BinOpKind::Shl, (16, 17)),
                Token::Shr => (BinOpKind::Shr, (16, 17)),
                Token::Plus => (BinOpKind::Add, (18, 19)),
                Token::Minus => (BinOpKind::Sub, (18, 19)),
                Token::Star => (BinOpKind::Mul, (20, 21)),
                Token::Slash => (BinOpKind::Div, (20, 21)),
                Token::Percent => (BinOpKind::Rem, (20, 21)),
                Token::As => {
                    if min_bp > 22 {
                        break;
                    }
                    let span = self.span();
                    self.advance(); // as
                    let ty = self.parse_type()?;
                    lhs = Expr::Cast {
                        expr: Box::new(lhs),
                        ty,
                        span,
                    };
                    continue;
                }
                _ => break,
            };

            let (l_bp, r_bp) = bp;
            if l_bp < min_bp {
                break;
            }

            let span = self.span();
            self.advance(); // consume operator
            let rhs = self.parse_precedence(r_bp)?;
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }

        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Token::Minus => {
                let span = self.span();
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Neg,
                    expr: Box::new(expr),
                    span,
                })
            }
            Token::Bang => {
                let span = self.span();
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOpKind::Not,
                    expr: Box::new(expr),
                    span,
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        // Handle .x, .y, .z field access
        while *self.peek() == Token::Dot {
            let span = self.span();
            self.advance(); // consume .
            let (field, _) = self.expect_ident()?;
            expr = Expr::FieldAccess {
                expr: Box::new(expr),
                field,
                span,
            };
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Token::FloatLit(v) => {
                let span = self.span();
                self.advance();
                Ok(Expr::FloatLit(v, span))
            }
            Token::IntLit(v) => {
                let span = self.span();
                self.advance();
                Ok(Expr::IntLit(v, span))
            }
            Token::U32Lit(v) => {
                let span = self.span();
                self.advance();
                Ok(Expr::U32Lit(v, span))
            }
            Token::True => {
                let span = self.span();
                self.advance();
                Ok(Expr::BoolLit(true, span))
            }
            Token::False => {
                let span = self.span();
                self.advance();
                Ok(Expr::BoolLit(false, span))
            }
            // vec2(...) and vec3(...) constructors — type keywords used as function calls
            Token::TyVec2 | Token::TyVec3 => {
                let name = match self.peek() {
                    Token::TyVec2 => "vec2".to_string(),
                    Token::TyVec3 => "vec3".to_string(),
                    _ => unreachable!(),
                };
                let span = self.span();
                self.advance();
                self.expect(&Token::LParen)?;
                let mut args = Vec::new();
                if *self.peek() != Token::RParen {
                    loop {
                        args.push(self.parse_expr()?);
                        if *self.peek() != Token::Comma {
                            break;
                        }
                        self.advance();
                        if *self.peek() == Token::RParen {
                            break;
                        }
                    }
                }
                self.expect(&Token::RParen)?;
                Ok(Expr::Call { name, args, span })
            }
            Token::Ident(name) => {
                let span = self.span();
                self.advance();
                // Check for function call
                if *self.peek() == Token::LParen {
                    self.advance(); // (
                    let mut args = Vec::new();
                    if *self.peek() != Token::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if *self.peek() != Token::Comma {
                                break;
                            }
                            self.advance();
                            // Allow trailing comma
                            if *self.peek() == Token::RParen {
                                break;
                            }
                        }
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call { name, args, span })
                } else {
                    Ok(Expr::Ident(name, span))
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::If => self.parse_if_expr(),
            _ => Err(self.error(format!("expected expression, got '{}'", self.peek()))),
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let span = self.span();
        self.advance(); // if
        let cond = self.parse_expr()?;
        self.expect(&Token::LBrace)?;
        let then_expr = self.parse_expr()?;
        self.expect(&Token::RBrace)?;
        self.expect(&Token::Else)?;
        let else_expr = if *self.peek() == Token::If {
            // else if — parse as nested if-expr (no braces needed around it)
            self.parse_if_expr()?
        } else {
            self.expect(&Token::LBrace)?;
            let e = self.parse_expr()?;
            self.expect(&Token::RBrace)?;
            e
        };
        Ok(Expr::IfElse {
            cond: Box::new(cond),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
            span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::lexer::lex;

    fn parse_expr_str(s: &str) -> Expr {
        let tokens = lex(s).unwrap();
        let mut parser = Parser::new(tokens);
        parser.parse_expr().unwrap()
    }

    #[test]
    fn precedence_mul_add() {
        let expr = parse_expr_str("a + b * c");
        // Should parse as a + (b * c)
        match expr {
            Expr::BinOp { op: BinOpKind::Add, rhs, .. } => {
                assert!(matches!(*rhs, Expr::BinOp { op: BinOpKind::Mul, .. }));
            }
            _ => panic!("expected add at top"),
        }
    }

    #[test]
    fn precedence_comparison() {
        let expr = parse_expr_str("a + b > c * d");
        assert!(matches!(expr, Expr::BinOp { op: BinOpKind::Gt, .. }));
    }

    #[test]
    fn function_call() {
        let expr = parse_expr_str("sqrt(x * x + y * y)");
        assert!(matches!(expr, Expr::Call { .. }));
    }

    #[test]
    fn cast_expr() {
        let expr = parse_expr_str("x as u32");
        assert!(matches!(expr, Expr::Cast { ty: ValType::U32, .. }));
    }

    #[test]
    fn if_else_expr() {
        let expr = parse_expr_str("if a > b { a } else { b }");
        assert!(matches!(expr, Expr::IfElse { .. }));
    }

    #[test]
    fn parse_kernel() {
        let src = r#"
            kernel test(x: f64, y: f64) -> u32 {
                let r = x * 255.0;
                emit r as u32;
            }
        "#;
        let tokens = lex(src).unwrap();
        let mut parser = Parser::new(tokens);
        let prog = parser.parse_program().unwrap();
        assert_eq!(prog.kernel.name, "test");
        assert_eq!(prog.kernel.params.len(), 2);
    }

    #[test]
    fn parse_while_loop() {
        let src = r#"
            kernel test(x: f64, y: f64) -> u32 {
                while zx = 0.0, zy = 0.0, iter: u32 = 0 {
                    break_if iter >= 256u32;
                    yield zx + 1.0, zy + 1.0, iter + 1u32;
                }
                emit iter;
            }
        "#;
        let tokens = lex(src).unwrap();
        let mut parser = Parser::new(tokens);
        let prog = parser.parse_program().unwrap();
        assert_eq!(prog.kernel.body.len(), 2); // while + emit
    }
}
