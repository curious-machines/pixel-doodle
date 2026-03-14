use crate::kernel_ir::*;
use std::collections::HashMap;

// ── Tokens ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Kernel,
    Emit,
    Const_,
    Select,
    PackArgb,
    While,
    Carry,
    Cond,
    Yield,
    // Types
    TyF64,
    TyU32,
    TyBool,
    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    And,
    Or,
    // Comparison ops
    Eq_,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Unary ops
    Neg,
    Not,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    // Conversion ops
    F64ToU32,
    U32ToF64,
    // Literals
    FloatLit(f64),
    IntLit(u64),
    True,
    False,
    // Punctuation
    LBrace,
    RBrace,
    LParen,
    RParen,
    Comma,
    Colon,
    Equals,
    // Identifier (anything not a keyword)
    Ident(String),
    // End of file
    Eof,
}

#[derive(Debug, Clone)]
struct Spanned {
    token: Token,
    line: usize,
    col: usize,
}

// ── Error ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
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

impl std::error::Error for ParseError {}

// ── Lexer ───────────────────────────────────────────────────────────

fn keyword_lookup(word: &str) -> Token {
    match word {
        "kernel" => Token::Kernel,
        "emit" => Token::Emit,
        "const" => Token::Const_,
        "select" => Token::Select,
        "pack_argb" => Token::PackArgb,
        "while" => Token::While,
        "carry" => Token::Carry,
        "cond" => Token::Cond,
        "yield" => Token::Yield,
        "f64" => Token::TyF64,
        "u32" => Token::TyU32,
        "bool" => Token::TyBool,
        "add" => Token::Add,
        "sub" => Token::Sub,
        "mul" => Token::Mul,
        "div" => Token::Div,
        "rem" => Token::Rem,
        "bit_and" => Token::BitAnd,
        "bit_or" => Token::BitOr,
        "bit_xor" => Token::BitXor,
        "shl" => Token::Shl,
        "shr" => Token::Shr,
        "and" => Token::And,
        "or" => Token::Or,
        "eq" => Token::Eq_,
        "ne" => Token::Ne,
        "lt" => Token::Lt,
        "le" => Token::Le,
        "gt" => Token::Gt,
        "ge" => Token::Ge,
        "neg" => Token::Neg,
        "not" => Token::Not,
        "abs" => Token::Abs,
        "sqrt" => Token::Sqrt,
        "floor" => Token::Floor,
        "ceil" => Token::Ceil,
        "f64_to_u32" => Token::F64ToU32,
        "u32_to_f64" => Token::U32ToF64,
        "true" => Token::True,
        "false" => Token::False,
        _ => Token::Ident(word.to_string()),
    }
}

fn lex(input: &str) -> Result<Vec<Spanned>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();
    let mut line = 1usize;
    let mut line_start = 0usize;

    while let Some(&(pos, ch)) = chars.peek() {
        let col = pos - line_start + 1;

        match ch {
            '\n' => {
                chars.next();
                line += 1;
                line_start = pos + 1;
            }
            c if c.is_ascii_whitespace() => {
                chars.next();
            }
            '#' => {
                // Line comment
                while let Some(&(_, c)) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '{' => {
                tokens.push(Spanned { token: Token::LBrace, line, col });
                chars.next();
            }
            '}' => {
                tokens.push(Spanned { token: Token::RBrace, line, col });
                chars.next();
            }
            '(' => {
                tokens.push(Spanned { token: Token::LParen, line, col });
                chars.next();
            }
            ')' => {
                tokens.push(Spanned { token: Token::RParen, line, col });
                chars.next();
            }
            ',' => {
                tokens.push(Spanned { token: Token::Comma, line, col });
                chars.next();
            }
            ':' => {
                tokens.push(Spanned { token: Token::Colon, line, col });
                chars.next();
            }
            '=' => {
                tokens.push(Spanned { token: Token::Equals, line, col });
                chars.next();
            }
            c if c.is_ascii_digit() || (c == '-' && matches!(chars.clone().nth(1), Some((_, d)) if d.is_ascii_digit())) => {
                let start = pos;
                let tok_col = col;
                if ch == '-' {
                    chars.next();
                }
                while let Some(&(_, d)) = chars.peek() {
                    if d.is_ascii_digit() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let mut is_float = false;
                if let Some(&(_, '.')) = chars.peek() {
                    // Check next char is a digit (not just a dot)
                    let mut lookahead = chars.clone();
                    lookahead.next(); // skip '.'
                    if let Some(&(_, d)) = lookahead.peek() {
                        if d.is_ascii_digit() {
                            is_float = true;
                            chars.next(); // consume '.'
                            while let Some(&(_, d)) = chars.peek() {
                                if d.is_ascii_digit() {
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
                let end = chars.peek().map_or(input.len(), |&(p, _)| p);
                let text = &input[start..end];
                if is_float {
                    let val: f64 = text.parse().map_err(|_| ParseError {
                        line,
                        col: tok_col,
                        message: format!("invalid float literal: {text}"),
                    })?;
                    tokens.push(Spanned { token: Token::FloatLit(val), line, col: tok_col });
                } else {
                    // Could be negative — parse as i64 first
                    let val: i64 = text.parse().map_err(|_| ParseError {
                        line,
                        col: tok_col,
                        message: format!("invalid integer literal: {text}"),
                    })?;
                    if val < 0 {
                        // Negative literal — treat as float for now since u32 can't be negative
                        // Actually, for our language negative int literals don't make sense for u32.
                        // We'll emit it as a float.
                        tokens.push(Spanned {
                            token: Token::FloatLit(val as f64),
                            line,
                            col: tok_col,
                        });
                    } else {
                        tokens.push(Spanned { token: Token::IntLit(val as u64), line, col: tok_col });
                    }
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = pos;
                let tok_col = col;
                while let Some(&(_, c)) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map_or(input.len(), |&(p, _)| p);
                let word = &input[start..end];
                tokens.push(Spanned { token: keyword_lookup(word), line, col: tok_col });
            }
            _ => {
                return Err(ParseError {
                    line,
                    col,
                    message: format!("unexpected character: '{ch}'"),
                });
            }
        }
    }

    tokens.push(Spanned { token: Token::Eof, line, col: 1 });
    Ok(tokens)
}

// ── Parser ──────────────────────────────────────────────────────────

/// Parse a .pdl source file into a Kernel IR.
pub fn parse(input: &str) -> Result<Kernel, ParseError> {
    let tokens = lex(input)?;
    let mut parser = Parser::new(tokens);
    parser.parse_kernel()
}

// ── Parser (cleaner approach with implicit const buffer) ───────────

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    vars: HashMap<String, Var>,
    var_types: HashMap<Var, ScalarType>,
    next_var: u32,
    implicit_stmts: Vec<Statement>,
}

impl Parser {
    fn new(tokens: Vec<Spanned>) -> Self {
        let mut vars = HashMap::new();
        let mut var_types = HashMap::new();
        vars.insert("x".to_string(), Var(0));
        vars.insert("y".to_string(), Var(1));
        var_types.insert(Var(0), ScalarType::F64);
        var_types.insert(Var(1), ScalarType::F64);
        Self {
            tokens,
            pos: 0,
            vars,
            var_types,
            next_var: 2,
            implicit_stmts: Vec::new(),
        }
    }

    fn peek(&self) -> &Spanned {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> Spanned {
        let t = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn expect_tok(&mut self, expected: &Token) -> Result<Spanned, ParseError> {
        let sp = self.peek().clone();
        if &sp.token == expected {
            self.advance();
            Ok(sp)
        } else {
            Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected {expected:?}, got {:?}", sp.token),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<(String, usize, usize), ParseError> {
        let sp = self.peek().clone();
        match sp.token {
            Token::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok((name, sp.line, sp.col))
            }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected identifier, got {:?}", sp.token),
            }),
        }
    }

    fn resolve_var(&self, name: &str, line: usize, col: usize) -> Result<Var, ParseError> {
        self.vars.get(name).copied().ok_or_else(|| ParseError {
            line,
            col,
            message: format!("undefined variable: '{name}'"),
        })
    }

    fn resolve_var_ident(&mut self) -> Result<Var, ParseError> {
        let (name, line, col) = self.expect_ident()?;
        self.resolve_var(&name, line, col)
    }

    fn alloc_var(&mut self, name: String, ty: ScalarType, line: usize, col: usize) -> Result<Var, ParseError> {
        if self.vars.contains_key(&name) {
            return Err(ParseError {
                line,
                col,
                message: format!("duplicate variable name: '{name}'"),
            });
        }
        let var = Var(self.next_var);
        self.next_var += 1;
        self.vars.insert(name, var);
        self.var_types.insert(var, ty);
        Ok(var)
    }

    fn alloc_implicit(&mut self, ty: ScalarType, const_val: Const) -> Var {
        let var = Var(self.next_var);
        self.next_var += 1;
        let name = format!("__lit_{}", var.0);
        self.vars.insert(name.clone(), var);
        self.var_types.insert(var, ty);
        self.implicit_stmts.push(Statement {
            binding: Binding { var, name, ty },
            inst: Inst::Const(const_val),
        });
        var
    }

    fn var_ty(&self, var: Var) -> ScalarType {
        self.var_types[&var]
    }

    fn parse_type(&mut self) -> Result<ScalarType, ParseError> {
        let sp = self.peek().clone();
        match sp.token {
            Token::TyF64 => { self.advance(); Ok(ScalarType::F64) }
            Token::TyU32 => { self.advance(); Ok(ScalarType::U32) }
            Token::TyBool => { self.advance(); Ok(ScalarType::Bool) }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected type (f64, u32, bool), got {:?}", sp.token),
            }),
        }
    }

    fn check_types_match(&self, expected: ScalarType, got: ScalarType, ctx: &str, line: usize, col: usize) -> Result<(), ParseError> {
        if expected != got {
            Err(ParseError {
                line,
                col,
                message: format!("type mismatch for {ctx}: expected {expected}, got {got}"),
            })
        } else {
            Ok(())
        }
    }

    /// Parse a variable reference OR an inline numeric/bool literal.
    /// If a literal is found, an implicit const statement is created.
    fn parse_operand(&mut self, expected_ty: ScalarType) -> Result<Var, ParseError> {
        let sp = self.peek().clone();
        match (&sp.token, expected_ty) {
            (Token::FloatLit(v), ScalarType::F64) => {
                let v = *v;
                self.advance();
                Ok(self.alloc_implicit(ScalarType::F64, Const::F64(v)))
            }
            (Token::IntLit(v), ScalarType::U32) => {
                let v = *v;
                self.advance();
                Ok(self.alloc_implicit(ScalarType::U32, Const::U32(v as u32)))
            }
            (Token::IntLit(v), ScalarType::F64) => {
                let v = *v as f64;
                self.advance();
                Ok(self.alloc_implicit(ScalarType::F64, Const::F64(v)))
            }
            (Token::True, ScalarType::Bool) => {
                self.advance();
                Ok(self.alloc_implicit(ScalarType::Bool, Const::Bool(true)))
            }
            (Token::False, ScalarType::Bool) => {
                self.advance();
                Ok(self.alloc_implicit(ScalarType::Bool, Const::Bool(false)))
            }
            _ => self.resolve_var_ident(),
        }
    }

    fn parse_instruction(&mut self, declared_ty: ScalarType, line: usize, col: usize) -> Result<Inst, ParseError> {
        let sp = self.peek().clone();
        match &sp.token {
            Token::Const_ => {
                self.advance();
                self.parse_const_literal(declared_ty)
            }
            Token::Add | Token::Sub | Token::Mul | Token::Div | Token::Rem => {
                let op = match sp.token {
                    Token::Add => BinOp::Add,
                    Token::Sub => BinOp::Sub,
                    Token::Mul => BinOp::Mul,
                    Token::Div => BinOp::Div,
                    Token::Rem => BinOp::Rem,
                    _ => unreachable!(),
                };
                self.advance();
                if declared_ty != ScalarType::F64 && declared_ty != ScalarType::U32 {
                    return Err(ParseError { line, col, message: format!("arithmetic ops require f64 or u32, got {declared_ty}") });
                }
                let lhs = self.parse_operand(declared_ty)?;
                self.check_types_match(declared_ty, self.var_ty(lhs), "lhs", line, col)?;
                let rhs = self.parse_operand(declared_ty)?;
                self.check_types_match(declared_ty, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Binary { op, lhs, rhs })
            }
            Token::BitAnd | Token::BitOr | Token::BitXor | Token::Shl | Token::Shr => {
                let op = match sp.token {
                    Token::BitAnd => BinOp::BitAnd,
                    Token::BitOr => BinOp::BitOr,
                    Token::BitXor => BinOp::BitXor,
                    Token::Shl => BinOp::Shl,
                    Token::Shr => BinOp::Shr,
                    _ => unreachable!(),
                };
                self.advance();
                if declared_ty != ScalarType::U32 {
                    return Err(ParseError { line, col, message: format!("bitwise ops require u32, got {declared_ty}") });
                }
                let lhs = self.parse_operand(ScalarType::U32)?;
                self.check_types_match(ScalarType::U32, self.var_ty(lhs), "lhs", line, col)?;
                let rhs = self.parse_operand(ScalarType::U32)?;
                self.check_types_match(ScalarType::U32, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Binary { op, lhs, rhs })
            }
            Token::And | Token::Or => {
                let op = if sp.token == Token::And { BinOp::And } else { BinOp::Or };
                self.advance();
                if declared_ty != ScalarType::Bool {
                    return Err(ParseError { line, col, message: format!("logical ops require bool, got {declared_ty}") });
                }
                let lhs = self.parse_operand(ScalarType::Bool)?;
                self.check_types_match(ScalarType::Bool, self.var_ty(lhs), "lhs", line, col)?;
                let rhs = self.parse_operand(ScalarType::Bool)?;
                self.check_types_match(ScalarType::Bool, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Binary { op, lhs, rhs })
            }
            Token::Eq_ | Token::Ne | Token::Lt | Token::Le | Token::Gt | Token::Ge => {
                let op = match sp.token {
                    Token::Eq_ => CmpOp::Eq,
                    Token::Ne => CmpOp::Ne,
                    Token::Lt => CmpOp::Lt,
                    Token::Le => CmpOp::Le,
                    Token::Gt => CmpOp::Gt,
                    Token::Ge => CmpOp::Ge,
                    _ => unreachable!(),
                };
                self.advance();
                if declared_ty != ScalarType::Bool {
                    return Err(ParseError { line, col, message: format!("comparison ops produce bool, got {declared_ty}") });
                }
                // For comparisons, we need to figure out the operand type from the first operand
                let lhs = self.resolve_var_ident()?;
                let operand_ty = self.var_ty(lhs);
                if operand_ty == ScalarType::Bool {
                    return Err(ParseError { line, col, message: "cannot compare bools".to_string() });
                }
                let rhs = self.parse_operand(operand_ty)?;
                self.check_types_match(operand_ty, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Cmp { op, lhs, rhs })
            }
            Token::Neg | Token::Abs => {
                let op = if sp.token == Token::Neg { UnaryOp::Neg } else { UnaryOp::Abs };
                self.advance();
                if declared_ty != ScalarType::F64 && declared_ty != ScalarType::U32 {
                    return Err(ParseError { line, col, message: format!("{op:?} requires f64 or u32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(declared_ty, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Unary { op, arg })
            }
            Token::Not => {
                self.advance();
                if declared_ty != ScalarType::Bool {
                    return Err(ParseError { line, col, message: format!("not requires bool, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ScalarType::Bool, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Unary { op: UnaryOp::Not, arg })
            }
            Token::Sqrt | Token::Floor | Token::Ceil => {
                let op = match sp.token {
                    Token::Sqrt => UnaryOp::Sqrt,
                    Token::Floor => UnaryOp::Floor,
                    Token::Ceil => UnaryOp::Ceil,
                    _ => unreachable!(),
                };
                self.advance();
                if declared_ty != ScalarType::F64 {
                    return Err(ParseError { line, col, message: format!("{op:?} requires f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ScalarType::F64, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Unary { op, arg })
            }
            Token::F64ToU32 => {
                self.advance();
                if declared_ty != ScalarType::U32 {
                    return Err(ParseError { line, col, message: format!("f64_to_u32 produces u32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ScalarType::F64, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F64ToU32, arg })
            }
            Token::U32ToF64 => {
                self.advance();
                if declared_ty != ScalarType::F64 {
                    return Err(ParseError { line, col, message: format!("u32_to_f64 produces f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ScalarType::U32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::U32ToF64, arg })
            }
            Token::Select => {
                self.advance();
                let cond = self.resolve_var_ident()?;
                self.check_types_match(ScalarType::Bool, self.var_ty(cond), "cond", line, col)?;
                let then_val = self.resolve_var_ident()?;
                let else_val = self.resolve_var_ident()?;
                let tty = self.var_ty(then_val);
                let ety = self.var_ty(else_val);
                if tty != ety {
                    return Err(ParseError { line, col, message: format!("select branches must be same type: {tty} vs {ety}") });
                }
                self.check_types_match(declared_ty, tty, "select result", line, col)?;
                Ok(Inst::Select { cond, then_val, else_val })
            }
            Token::PackArgb => {
                self.advance();
                if declared_ty != ScalarType::U32 {
                    return Err(ParseError { line, col, message: format!("pack_argb produces u32, got {declared_ty}") });
                }
                let r = self.parse_operand(ScalarType::U32)?;
                self.check_types_match(ScalarType::U32, self.var_ty(r), "r", line, col)?;
                let g = self.parse_operand(ScalarType::U32)?;
                self.check_types_match(ScalarType::U32, self.var_ty(g), "g", line, col)?;
                let b = self.parse_operand(ScalarType::U32)?;
                self.check_types_match(ScalarType::U32, self.var_ty(b), "b", line, col)?;
                Ok(Inst::PackArgb { r, g, b })
            }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected instruction, got {:?}", sp.token),
            }),
        }
    }

    fn parse_const_literal(&mut self, declared_ty: ScalarType) -> Result<Inst, ParseError> {
        let sp = self.peek().clone();
        match (&sp.token, declared_ty) {
            (Token::FloatLit(v), ScalarType::F64) => {
                let v = *v;
                self.advance();
                Ok(Inst::Const(Const::F64(v)))
            }
            (Token::IntLit(v), ScalarType::U32) => {
                let v = *v;
                self.advance();
                if v > u32::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("u32 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::U32(v as u32)))
            }
            (Token::IntLit(v), ScalarType::F64) => {
                let v = *v as f64;
                self.advance();
                Ok(Inst::Const(Const::F64(v)))
            }
            (Token::True, ScalarType::Bool) => {
                self.advance();
                Ok(Inst::Const(Const::Bool(true)))
            }
            (Token::False, ScalarType::Bool) => {
                self.advance();
                Ok(Inst::Const(Const::Bool(false)))
            }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("const literal type mismatch: expected {declared_ty}, got {:?}", sp.token),
            }),
        }
    }

    /// Parse a single statement: `name: type = instruction`
    fn parse_statement(&mut self) -> Result<(Statement, Vec<Statement>), ParseError> {
        let (var_name, line, col) = self.expect_ident()?;
        self.expect_tok(&Token::Colon)?;
        let ty = self.parse_type()?;
        self.expect_tok(&Token::Equals)?;

        self.implicit_stmts.clear();
        let inst = self.parse_instruction(ty, line, col)?;

        let implicits = std::mem::take(&mut self.implicit_stmts);
        let var = self.alloc_var(var_name.clone(), ty, line, col)?;
        let stmt = Statement {
            binding: Binding { var, name: var_name, ty },
            inst,
        };
        Ok((stmt, implicits))
    }

    /// Parse a while loop: `while carry(name: type = init, ...) { ... cond var ... yield v1 v2 ... }`
    fn parse_while(&mut self) -> Result<While, ParseError> {
        let while_sp = self.expect_tok(&Token::While)?;
        self.expect_tok(&Token::Carry)?;
        self.expect_tok(&Token::LParen)?;

        // Save outer scope for cleanup
        let outer_vars: HashMap<String, Var> = self.vars.clone();

        // Parse carry variable declarations
        let mut carry = Vec::new();
        loop {
            let sp = self.peek().clone();
            if sp.token == Token::RParen {
                self.advance();
                break;
            }
            if !carry.is_empty() {
                self.expect_tok(&Token::Comma)?;
            }
            let (name, line, col) = self.expect_ident()?;
            self.expect_tok(&Token::Colon)?;
            let ty = self.parse_type()?;
            self.expect_tok(&Token::Equals)?;
            let init = self.resolve_var_ident()?;
            self.check_types_match(ty, self.var_ty(init), &format!("carry var '{name}' init"), line, col)?;

            let var = self.alloc_var(name.clone(), ty, line, col)?;
            carry.push(CarryVar {
                binding: Binding { var, name, ty },
                init,
            });
        }

        if carry.is_empty() {
            return Err(ParseError {
                line: while_sp.line,
                col: while_sp.col,
                message: "while loop must have at least one carry variable".to_string(),
            });
        }

        self.expect_tok(&Token::LBrace)?;

        // Parse cond_body: statements until `cond`
        let mut cond_body = Vec::new();
        loop {
            let sp = self.peek().clone();
            match sp.token {
                Token::Cond => break,
                Token::RBrace | Token::Eof => {
                    return Err(ParseError {
                        line: sp.line,
                        col: sp.col,
                        message: "expected 'cond' in while body".to_string(),
                    });
                }
                _ => {
                    let (stmt, implicits) = self.parse_statement()?;
                    for imp in implicits {
                        cond_body.push(imp);
                    }
                    cond_body.push(stmt);
                }
            }
        }

        // Parse `cond <var>`
        self.expect_tok(&Token::Cond)?;
        let (cond_name, cond_line, cond_col) = self.expect_ident()?;
        let cond = self.resolve_var(&cond_name, cond_line, cond_col)?;
        self.check_types_match(ScalarType::Bool, self.var_ty(cond), "cond", cond_line, cond_col)?;

        // Parse body: statements until `yield`
        let mut body = Vec::new();
        loop {
            let sp = self.peek().clone();
            match sp.token {
                Token::Yield => break,
                Token::RBrace | Token::Eof => {
                    return Err(ParseError {
                        line: sp.line,
                        col: sp.col,
                        message: "expected 'yield' in while body".to_string(),
                    });
                }
                _ => {
                    let (stmt, implicits) = self.parse_statement()?;
                    for imp in implicits {
                        body.push(imp);
                    }
                    body.push(stmt);
                }
            }
        }

        // Parse `yield v1 v2 ...`
        let yield_sp = self.expect_tok(&Token::Yield)?;
        let mut yields = Vec::new();
        for (i, cv) in carry.iter().enumerate() {
            let (name, line, col) = self.expect_ident()?;
            let var = self.resolve_var(&name, line, col)?;
            self.check_types_match(cv.binding.ty, self.var_ty(var),
                &format!("yield[{}] for carry var '{}'", i, cv.binding.name), line, col)?;
            yields.push(var);
        }

        if yields.len() != carry.len() {
            return Err(ParseError {
                line: yield_sp.line,
                col: yield_sp.col,
                message: format!("yield has {} values but {} carry variables", yields.len(), carry.len()),
            });
        }

        self.expect_tok(&Token::RBrace)?;

        // Restore outer scope but keep carry vars visible
        let carry_entries: Vec<(String, Var, ScalarType)> = carry.iter()
            .map(|cv| (cv.binding.name.clone(), cv.binding.var, cv.binding.ty))
            .collect();
        self.vars = outer_vars;
        for (name, var, ty) in carry_entries {
            self.vars.insert(name, var);
            self.var_types.insert(var, ty);
        }

        Ok(While { carry, cond_body, cond, body, yields })
    }

    fn parse_kernel(&mut self) -> Result<Kernel, ParseError> {
        self.expect_tok(&Token::Kernel)?;
        let (name, _line, _col) = self.expect_ident()?;
        self.expect_tok(&Token::LBrace)?;

        let mut body = Vec::new();

        loop {
            let sp = self.peek().clone();
            match &sp.token {
                Token::Emit => break,
                Token::RBrace => {
                    return Err(ParseError {
                        line: sp.line,
                        col: sp.col,
                        message: "expected 'emit' before closing brace".to_string(),
                    });
                }
                Token::Eof => {
                    return Err(ParseError {
                        line: sp.line,
                        col: sp.col,
                        message: "unexpected end of file, expected 'emit'".to_string(),
                    });
                }
                Token::While => {
                    let w = self.parse_while()?;
                    body.push(BodyItem::While(w));
                }
                _ => {
                    let (stmt, implicits) = self.parse_statement()?;
                    for imp in implicits {
                        body.push(BodyItem::Stmt(imp));
                    }
                    body.push(BodyItem::Stmt(stmt));
                }
            }
        }

        self.expect_tok(&Token::Emit)?;
        let (emit_name, emit_line, emit_col) = self.expect_ident()?;
        let emit_var = self.resolve_var(&emit_name, emit_line, emit_col)?;
        if self.var_ty(emit_var) != ScalarType::U32 {
            return Err(ParseError {
                line: emit_line,
                col: emit_col,
                message: format!("emit must reference a u32 variable, got {}", self.var_ty(emit_var)),
            });
        }

        self.expect_tok(&Token::RBrace)?;

        Ok(Kernel { name, body, emit: emit_var })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gradient() {
        let src = r#"
kernel gradient {
    r: f64 = mul x 255.0
    r_u: u32 = f64_to_u32 r
    g: f64 = mul y 255.0
    g_u: u32 = f64_to_u32 g
    b: u32 = const 128
    pixel: u32 = pack_argb r_u g_u b
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "gradient");
        // x=Var(0), y=Var(1), implicit 255.0=Var(2), r=Var(3),
        // r_u=Var(4), implicit 255.0=Var(5), g=Var(6), g_u=Var(7),
        // b=Var(8), pixel=Var(9)
        assert_eq!(kernel.emit, Var(9));
    }

    #[test]
    fn test_parse_solid_color() {
        let src = r#"
kernel solid {
    r: u32 = const 255
    g: u32 = const 0
    b: u32 = const 128
    pixel: u32 = pack_argb r g b
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "solid");
    }

    #[test]
    fn test_parse_error_undefined_var() {
        let src = r#"
kernel bad {
    r: u32 = const 255
    pixel: u32 = pack_argb r g b
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("undefined variable"), "got: {}", err.message);
    }

    #[test]
    fn test_parse_error_type_mismatch() {
        let src = r#"
kernel bad {
    r: f64 = mul x 255.0
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("type mismatch"), "got: {}", err.message);
    }

    #[test]
    fn test_parse_error_duplicate_var() {
        let src = r#"
kernel bad {
    r: u32 = const 255
    r: u32 = const 0
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("duplicate"), "got: {}", err.message);
    }

    #[test]
    fn test_roundtrip() {
        let src = r#"kernel gradient {
    r: f64 = mul x 255.0
    r_u: u32 = f64_to_u32 r
    g: f64 = mul y 255.0
    g_u: u32 = f64_to_u32 g
    b: u32 = const 128
    pixel: u32 = pack_argb r_u g_u b
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        let printed = crate::lang::printer::print(&kernel);
        let kernel2 = parse(&printed).unwrap();
        assert_eq!(kernel2.name, kernel.name);
        assert_eq!(kernel2.body.len(), kernel.body.len());
    }

    #[test]
    fn test_parse_while_basic() {
        let src = r#"
kernel loop_test {
    zero: f64 = const 0.0
    limit: f64 = const 10.0
    init_i: u32 = const 0
    while carry(val: f64 = zero, i: u32 = init_i) {
        done: bool = ge val limit
        cont: bool = not done
        cond cont
        new_val: f64 = add val 1.0
        new_i: u32 = add i 1
        yield new_val new_i
    }
    result: u32 = f64_to_u32 val
    pixel: u32 = pack_argb result result result
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "loop_test");
        // Check that while body item exists
        let has_while = kernel.body.iter().any(|item| matches!(item, BodyItem::While(_)));
        assert!(has_while, "expected a while body item");
    }

    #[test]
    fn test_parse_while_carry_vars_live_after() {
        // Carry vars (val, i) should be accessible after the while
        let src = r#"
kernel carry_test {
    zero: f64 = const 0.0
    init_i: u32 = const 0
    while carry(val: f64 = zero, i: u32 = init_i) {
        done: bool = ge i 5
        cont: bool = not done
        cond cont
        new_val: f64 = add val 1.0
        new_i: u32 = add i 1
        yield new_val new_i
    }
    # val and i should be in scope here
    result: u32 = f64_to_u32 val
    pixel: u32 = pack_argb result i result
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "carry_test");
    }

    #[test]
    fn test_parse_mandelbrot() {
        let src = include_str!("../../examples/mandelbrot.pdl");
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "mandelbrot");
        // Verify roundtrip
        let printed = crate::lang::printer::print(&kernel);
        let kernel2 = parse(&printed).unwrap();
        assert_eq!(kernel2.name, kernel.name);
    }

    #[test]
    fn test_parse_while_inner_vars_not_visible_outside() {
        // Variables defined inside the while body should not be visible after it
        let src = r#"
kernel scope_test {
    zero: f64 = const 0.0
    init_i: u32 = const 0
    while carry(val: f64 = zero, i: u32 = init_i) {
        done: bool = ge i 5
        cont: bool = not done
        cond cont
        new_val: f64 = add val 1.0
        new_i: u32 = add i 1
        yield new_val new_i
    }
    # 'done' and 'cont' should NOT be in scope here
    oops: bool = not done
    pixel: u32 = pack_argb i i i
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("undefined variable"), "got: {}", err.message);
    }
}
