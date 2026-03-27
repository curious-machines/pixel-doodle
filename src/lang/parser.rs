use crate::kernel_ir::{ScalarType, *};
use std::collections::HashMap;

// ── Tokens ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Kernel,
    Emit,
    Inline,
    Return,
    Const_,
    Select,
    PackArgb,
    While,
    Carry,
    Cond,
    Yield,
    // Types
    TyF32,
    TyF64,
    TyI8,
    TyU8,
    TyI16,
    TyU16,
    TyI32,
    TyU32,
    TyI64,
    TyU64,
    TyBool,
    TyVec2,
    TyVec3,
    TyVec4,
    TyMat2,
    TyMat3,
    TyMat4,
    // Angle brackets (for parameterized types like vec2<f64>)
    LAngle,
    RAngle,
    // Vec construction
    MakeVec2,
    MakeVec3,
    MakeVec4,
    // Vec extraction
    ExtractX,
    ExtractY,
    ExtractZ,
    ExtractW,
    // Vec binary ops
    VecAdd,
    VecSub,
    VecMul,
    VecDiv,
    VecMin,
    VecMax,
    // Vec unary ops
    VecNeg,
    VecAbs,
    VecNormalize,
    // Vec other
    VecScale,
    VecDot,
    VecLength,
    VecCross,
    // Mat construction & ops
    MakeMat2,
    MakeMat3,
    MakeMat4,
    MatMulVec,
    MatMul,
    MatTranspose,
    MatCol0,
    MatCol1,
    MatCol2,
    MatCol3,
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
    Min,
    Max,
    Atan2,
    Pow,
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
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Round,
    Trunc,
    Fract,
    // Conversion ops
    F64ToU32,
    U32ToF64,
    U32ToF64Norm,
    F32ToF64,
    F64ToF32,
    I32ToU32,
    U32ToI32,
    I32ToF64,
    F64ToI32,
    I32ToF32,
    F32ToI32,
    F32ToU32,
    U32ToF32,
    // Hash op (binary u32 -> u32)
    Hash,
    // Buffer ops
    BufLoad,
    BufStore,
    Buffers,
    Read_,
    Write_,
    // Literals
    FloatLit(f64),
    IntLit(u64),
    True,
    False,
    // Punctuation
    Arrow, // ->
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
        "inline" => Token::Inline,
        "return" => Token::Return,
        "const" => Token::Const_,
        "select" => Token::Select,
        "pack_argb" => Token::PackArgb,
        "while" => Token::While,
        "carry" => Token::Carry,
        "cond" => Token::Cond,
        "yield" => Token::Yield,
        "f32" => Token::TyF32,
        "f64" => Token::TyF64,
        "i8" => Token::TyI8,
        "u8" => Token::TyU8,
        "i16" => Token::TyI16,
        "u16" => Token::TyU16,
        "i32" => Token::TyI32,
        "u32" => Token::TyU32,
        "i64" => Token::TyI64,
        "u64" => Token::TyU64,
        "bool" => Token::TyBool,
        "vec2" => Token::TyVec2,
        "vec3" => Token::TyVec3,
        "vec4" => Token::TyVec4,
        "mat2" => Token::TyMat2,
        "mat3" => Token::TyMat3,
        "mat4" => Token::TyMat4,
        "make_vec2" => Token::MakeVec2,
        "make_vec3" => Token::MakeVec3,
        "make_vec4" => Token::MakeVec4,
        "extract_x" => Token::ExtractX,
        "extract_y" => Token::ExtractY,
        "extract_z" => Token::ExtractZ,
        "extract_w" => Token::ExtractW,
        "vec_add" => Token::VecAdd,
        "vec_sub" => Token::VecSub,
        "vec_mul" => Token::VecMul,
        "vec_div" => Token::VecDiv,
        "vec_min" => Token::VecMin,
        "vec_max" => Token::VecMax,
        "vec_neg" => Token::VecNeg,
        "vec_abs" => Token::VecAbs,
        "vec_normalize" => Token::VecNormalize,
        "vec_scale" => Token::VecScale,
        "vec_dot" => Token::VecDot,
        "vec_length" => Token::VecLength,
        "vec_cross" => Token::VecCross,
        "make_mat2" => Token::MakeMat2,
        "make_mat3" => Token::MakeMat3,
        "make_mat4" => Token::MakeMat4,
        "mat_mul_vec" => Token::MatMulVec,
        "mat_mul" => Token::MatMul,
        "mat_transpose" => Token::MatTranspose,
        "mat_col0" => Token::MatCol0,
        "mat_col1" => Token::MatCol1,
        "mat_col2" => Token::MatCol2,
        "mat_col3" => Token::MatCol3,
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
        "min" => Token::Min,
        "max" => Token::Max,
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
        "sin" => Token::Sin,
        "cos" => Token::Cos,
        "tan" => Token::Tan,
        "asin" => Token::Asin,
        "acos" => Token::Acos,
        "atan" => Token::Atan,
        "atan2" => Token::Atan2,
        "exp" => Token::Exp,
        "exp2" => Token::Exp2,
        "log" => Token::Log,
        "log2" => Token::Log2,
        "log10" => Token::Log10,
        "pow" => Token::Pow,
        "round" => Token::Round,
        "trunc" => Token::Trunc,
        "fract" => Token::Fract,
        "f64_to_u32" => Token::F64ToU32,
        "u32_to_f64" => Token::U32ToF64,
        "u32_to_f64_norm" => Token::U32ToF64Norm,
        "f32_to_f64" => Token::F32ToF64,
        "f64_to_f32" => Token::F64ToF32,
        "i32_to_u32" => Token::I32ToU32,
        "u32_to_i32" => Token::U32ToI32,
        "i32_to_f64" => Token::I32ToF64,
        "f64_to_i32" => Token::F64ToI32,
        "i32_to_f32" => Token::I32ToF32,
        "f32_to_i32" => Token::F32ToI32,
        "f32_to_u32" => Token::F32ToU32,
        "u32_to_f32" => Token::U32ToF32,
        "hash" => Token::Hash,
        "buf_load" => Token::BufLoad,
        "buf_store" => Token::BufStore,
        "buffers" => Token::Buffers,
        "read" => Token::Read_,
        "write" => Token::Write_,
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
            '<' => {
                tokens.push(Spanned { token: Token::LAngle, line, col });
                chars.next();
            }
            '>' => {
                tokens.push(Spanned { token: Token::RAngle, line, col });
                chars.next();
            }
            '-' if matches!(chars.clone().nth(1), Some((_, '>'))) => {
                tokens.push(Spanned { token: Token::Arrow, line, col });
                chars.next(); // consume '-'
                chars.next(); // consume '>'
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

/// Parse a .pdir source file into a Kernel IR.
pub fn parse(input: &str) -> Result<Kernel, ParseError> {
    let tokens = lex(input)?;
    let mut parser = Parser::new(tokens);
    // Parse any inline function definitions before the kernel
    while parser.peek().token == Token::Inline {
        parser.parse_inline_def()?;
    }
    parser.parse_kernel()
}

// ── Inline function definitions (parser-internal) ──────────────────

/// A pre-parsed inline function body item (stmt or while).
#[derive(Debug, Clone)]
enum InlineBodyItem {
    Stmt {
        name: String,
        ty: ValType,
        var: Var,
        inst: Inst,
    },
    While {
        w: While,
        /// Names of carry vars in declaration order (for prefixing during expansion).
        carry_names: Vec<String>,
        /// Names of all statements in cond_body and body (for prefixing).
        cond_body_names: Vec<String>,
        body_names: Vec<String>,
    },
}

/// Recursively collect names from body items (stmts + nested while carry/cond_body/body).
fn collect_body_item_names(items: &[BodyItem]) -> Vec<String> {
    let mut names = Vec::new();
    for item in items {
        match item {
            BodyItem::Stmt(stmt) => names.push(stmt.binding.name.clone()),
            BodyItem::While(w) => {
                for cv in &w.carry {
                    names.push(cv.binding.name.clone());
                }
                names.extend(collect_body_item_names(&w.cond_body));
                names.extend(collect_body_item_names(&w.body));
            }
        }
    }
    names
}

/// Count the number of names that `collect_body_item_names` would produce.
fn count_body_item_names(items: &[BodyItem]) -> usize {
    let mut count = 0;
    for item in items {
        match item {
            BodyItem::Stmt(_) => count += 1,
            BodyItem::While(w) => {
                count += w.carry.len();
                count += count_body_item_names(&w.cond_body);
                count += count_body_item_names(&w.body);
            }
        }
    }
    count
}

/// A parsed inline function definition, ready for expansion at call sites.
#[derive(Debug, Clone)]
struct InlineDef {
    params: Vec<(String, Var, ValType)>,
    return_ty: ValType,
    /// The instruction for the return value (extracted from the body).
    /// Only valid when `return_var_is_stmt` is true.
    return_inst: Inst,
    /// The original Var for the return value (before remapping).
    return_var: Var,
    /// True if return var was a body statement (extracted into return_inst).
    /// False if return var is a while carry var (expand_inline handles specially).
    return_var_is_stmt: bool,
    body: Vec<InlineBodyItem>,
}

// ── Parser (cleaner approach with implicit const buffer) ───────────

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    vars: HashMap<String, Var>,
    var_types: HashMap<Var, ValType>,
    next_var: u32,
    implicit_stmts: Vec<Statement>,
    inline_defs: HashMap<String, InlineDef>,
    inline_call_count: u32,
    /// Buffer for BodyItems produced during inline expansion (e.g. While blocks).
    expanded_body_items: Vec<BodyItem>,
    /// Buffer declarations for simulation kernels.
    buf_decls: Vec<BufDecl>,
    /// Map buffer names to indices.
    buf_names: HashMap<String, u32>,
}

impl Parser {
    fn new(tokens: Vec<Spanned>) -> Self {
        Self {
            tokens,
            pos: 0,
            vars: HashMap::new(),
            var_types: HashMap::new(),
            next_var: 0,
            implicit_stmts: Vec::new(),
            inline_defs: HashMap::new(),
            inline_call_count: 0,
            expanded_body_items: Vec::new(),
            buf_decls: Vec::new(),
            buf_names: HashMap::new(),
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

    fn alloc_var(&mut self, name: String, ty: ValType, line: usize, col: usize) -> Result<Var, ParseError> {
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

    fn alloc_implicit(&mut self, ty: ValType, const_val: Const) -> Var {
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

    fn var_ty(&self, var: Var) -> ValType {
        self.var_types[&var]
    }

    fn parse_scalar_type(&mut self) -> Result<ScalarType, ParseError> {
        let sp = self.peek().clone();
        match sp.token {
            Token::TyF32 => { self.advance(); Ok(ScalarType::F32) }
            Token::TyF64 => { self.advance(); Ok(ScalarType::F64) }
            Token::TyI8 => { self.advance(); Ok(ScalarType::I8) }
            Token::TyU8 => { self.advance(); Ok(ScalarType::U8) }
            Token::TyI16 => { self.advance(); Ok(ScalarType::I16) }
            Token::TyU16 => { self.advance(); Ok(ScalarType::U16) }
            Token::TyI32 => { self.advance(); Ok(ScalarType::I32) }
            Token::TyU32 => { self.advance(); Ok(ScalarType::U32) }
            Token::TyI64 => { self.advance(); Ok(ScalarType::I64) }
            Token::TyU64 => { self.advance(); Ok(ScalarType::U64) }
            Token::TyBool => { self.advance(); Ok(ScalarType::Bool) }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected scalar type, got {:?}", sp.token),
            }),
        }
    }

    fn parse_type(&mut self) -> Result<ValType, ParseError> {
        let sp = self.peek().clone();
        match sp.token {
            Token::TyF32 => { self.advance(); Ok(ValType::F32) }
            Token::TyF64 => { self.advance(); Ok(ValType::F64) }
            Token::TyI8 => { self.advance(); Ok(ValType::I8) }
            Token::TyU8 => { self.advance(); Ok(ValType::U8) }
            Token::TyI16 => { self.advance(); Ok(ValType::I16) }
            Token::TyU16 => { self.advance(); Ok(ValType::U16) }
            Token::TyI32 => { self.advance(); Ok(ValType::I32) }
            Token::TyU32 => { self.advance(); Ok(ValType::U32) }
            Token::TyI64 => { self.advance(); Ok(ValType::I64) }
            Token::TyU64 => { self.advance(); Ok(ValType::U64) }
            Token::TyBool => { self.advance(); Ok(ValType::BOOL) }
            Token::TyVec2 | Token::TyVec3 | Token::TyVec4 => {
                let len: u8 = match sp.token {
                    Token::TyVec2 => 2,
                    Token::TyVec3 => 3,
                    Token::TyVec4 => 4,
                    _ => unreachable!(),
                };
                self.advance();
                self.expect_tok(&Token::LAngle)?;
                let elem = self.parse_scalar_type()?;
                self.expect_tok(&Token::RAngle)?;
                Ok(ValType::Vec { len, elem })
            }
            Token::TyMat2 | Token::TyMat3 | Token::TyMat4 => {
                let size: u8 = match sp.token {
                    Token::TyMat2 => 2,
                    Token::TyMat3 => 3,
                    Token::TyMat4 => 4,
                    _ => unreachable!(),
                };
                self.advance();
                self.expect_tok(&Token::LAngle)?;
                let elem = self.parse_scalar_type()?;
                self.expect_tok(&Token::RAngle)?;
                Ok(ValType::Mat { size, elem })
            }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected type, got {:?}", sp.token),
            }),
        }
    }

    fn check_types_match(&self, expected: ValType, got: ValType, ctx: &str, line: usize, col: usize) -> Result<(), ParseError> {
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
    fn parse_operand(&mut self, expected_ty: ValType) -> Result<Var, ParseError> {
        let sp = self.peek().clone();
        match (&sp.token, expected_ty) {
            (Token::FloatLit(v), ValType::Scalar(ScalarType::F64)) => {
                let v = *v;
                self.advance();
                // Check for f32 suffix — if present, this is an error (expected f64)
                Ok(self.alloc_implicit(ValType::F64, Const::F64(v)))
            }
            (Token::FloatLit(v), ValType::Scalar(ScalarType::F32)) => {
                let v = *v as f32;
                self.advance();
                // Consume optional f32 suffix
                if self.peek().token == Token::TyF32 {
                    self.advance();
                }
                Ok(self.alloc_implicit(ValType::F32, Const::F32(v)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I8)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI8 { self.advance(); }
                Ok(self.alloc_implicit(ValType::I8, Const::I8(v as i8)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U8)) => {
                let v = *v;
                self.advance();
                Ok(self.alloc_implicit(ValType::U8, Const::U8(v as u8)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I16)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI16 { self.advance(); }
                Ok(self.alloc_implicit(ValType::I16, Const::I16(v as i16)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U16)) => {
                let v = *v;
                self.advance();
                Ok(self.alloc_implicit(ValType::U16, Const::U16(v as u16)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U32)) => {
                let v = *v;
                self.advance();
                Ok(self.alloc_implicit(ValType::U32, Const::U32(v as u32)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I32)) => {
                let v = *v;
                self.advance();
                // Consume optional i32 suffix
                if self.peek().token == Token::TyI32 {
                    self.advance();
                }
                Ok(self.alloc_implicit(ValType::I32, Const::I32(v as i32)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I64)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI64 { self.advance(); }
                Ok(self.alloc_implicit(ValType::I64, Const::I64(v as i64)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U64)) => {
                let v = *v;
                self.advance();
                Ok(self.alloc_implicit(ValType::U64, Const::U64(v)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::F64)) => {
                let v = *v as f64;
                self.advance();
                Ok(self.alloc_implicit(ValType::F64, Const::F64(v)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::F32)) => {
                let v = *v as f32;
                self.advance();
                // Consume optional f32 suffix
                if self.peek().token == Token::TyF32 {
                    self.advance();
                }
                Ok(self.alloc_implicit(ValType::F32, Const::F32(v)))
            }
            (Token::True, ValType::Scalar(ScalarType::Bool)) => {
                self.advance();
                Ok(self.alloc_implicit(ValType::BOOL, Const::Bool(true)))
            }
            (Token::False, ValType::Scalar(ScalarType::Bool)) => {
                self.advance();
                Ok(self.alloc_implicit(ValType::BOOL, Const::Bool(false)))
            }
            _ => self.resolve_var_ident(),
        }
    }

    fn parse_instruction(&mut self, declared_ty: ValType, line: usize, col: usize) -> Result<Inst, ParseError> {
        let sp = self.peek().clone();
        match &sp.token {
            Token::Const_ => {
                self.advance();
                self.parse_const_literal(declared_ty)
            }
            Token::Add | Token::Sub | Token::Mul | Token::Div | Token::Rem | Token::Min | Token::Max | Token::Atan2 | Token::Pow => {
                let op = match sp.token {
                    Token::Add => BinOp::Add,
                    Token::Sub => BinOp::Sub,
                    Token::Mul => BinOp::Mul,
                    Token::Div => BinOp::Div,
                    Token::Rem => BinOp::Rem,
                    Token::Min => BinOp::Min,
                    Token::Max => BinOp::Max,
                    Token::Atan2 => BinOp::Atan2,
                    Token::Pow => BinOp::Pow,
                    _ => unreachable!(),
                };
                self.advance();
                if !matches!(declared_ty, ValType::Scalar(s) if s.is_float() || s.is_integer()) {
                    return Err(ParseError { line, col, message: format!("arithmetic ops require numeric type, got {declared_ty}") });
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
                if !matches!(declared_ty, ValType::Scalar(s) if s.is_integer()) {
                    return Err(ParseError { line, col, message: format!("bitwise ops require integer type, got {declared_ty}") });
                }
                let lhs = self.parse_operand(declared_ty)?;
                self.check_types_match(declared_ty, self.var_ty(lhs), "lhs", line, col)?;
                let rhs = self.parse_operand(declared_ty)?;
                self.check_types_match(declared_ty, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Binary { op, lhs, rhs })
            }
            Token::And | Token::Or => {
                let op = if sp.token == Token::And { BinOp::And } else { BinOp::Or };
                self.advance();
                if declared_ty != ValType::BOOL {
                    return Err(ParseError { line, col, message: format!("logical ops require bool, got {declared_ty}") });
                }
                let lhs = self.parse_operand(ValType::BOOL)?;
                self.check_types_match(ValType::BOOL, self.var_ty(lhs), "lhs", line, col)?;
                let rhs = self.parse_operand(ValType::BOOL)?;
                self.check_types_match(ValType::BOOL, self.var_ty(rhs), "rhs", line, col)?;
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
                if declared_ty != ValType::BOOL {
                    return Err(ParseError { line, col, message: format!("comparison ops produce bool, got {declared_ty}") });
                }
                // For comparisons, we need to figure out the operand type from the first operand
                let lhs = self.resolve_var_ident()?;
                let operand_ty = self.var_ty(lhs);
                if operand_ty == ValType::BOOL {
                    return Err(ParseError { line, col, message: "cannot compare bools".to_string() });
                }
                let rhs = self.parse_operand(operand_ty)?;
                self.check_types_match(operand_ty, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Cmp { op, lhs, rhs })
            }
            Token::Neg | Token::Abs => {
                let op = if sp.token == Token::Neg { UnaryOp::Neg } else { UnaryOp::Abs };
                self.advance();
                if !matches!(declared_ty, ValType::Scalar(s) if s.is_float() || s.is_integer()) {
                    return Err(ParseError { line, col, message: format!("{op:?} requires numeric type, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(declared_ty, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Unary { op, arg })
            }
            Token::Not => {
                self.advance();
                if declared_ty != ValType::BOOL {
                    return Err(ParseError { line, col, message: format!("not requires bool, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::BOOL, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Unary { op: UnaryOp::Not, arg })
            }
            Token::Sqrt | Token::Floor | Token::Ceil
            | Token::Sin | Token::Cos | Token::Tan
            | Token::Asin | Token::Acos | Token::Atan
            | Token::Exp | Token::Exp2
            | Token::Log | Token::Log2 | Token::Log10
            | Token::Round | Token::Trunc | Token::Fract => {
                let op = match sp.token {
                    Token::Sqrt => UnaryOp::Sqrt,
                    Token::Floor => UnaryOp::Floor,
                    Token::Ceil => UnaryOp::Ceil,
                    Token::Sin => UnaryOp::Sin,
                    Token::Cos => UnaryOp::Cos,
                    Token::Tan => UnaryOp::Tan,
                    Token::Asin => UnaryOp::Asin,
                    Token::Acos => UnaryOp::Acos,
                    Token::Atan => UnaryOp::Atan,
                    Token::Exp => UnaryOp::Exp,
                    Token::Exp2 => UnaryOp::Exp2,
                    Token::Log => UnaryOp::Log,
                    Token::Log2 => UnaryOp::Log2,
                    Token::Log10 => UnaryOp::Log10,
                    Token::Round => UnaryOp::Round,
                    Token::Trunc => UnaryOp::Trunc,
                    Token::Fract => UnaryOp::Fract,
                    _ => unreachable!(),
                };
                self.advance();
                if !matches!(declared_ty, ValType::Scalar(ScalarType::F32) | ValType::Scalar(ScalarType::F64)) {
                    return Err(ParseError { line, col, message: format!("{op:?} requires f32 or f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(declared_ty, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Unary { op, arg })
            }
            Token::F64ToU32 => {
                self.advance();
                if declared_ty != ValType::U32 {
                    return Err(ParseError { line, col, message: format!("f64_to_u32 produces u32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::F64, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F64_TO_U32, arg })
            }
            Token::U32ToF64 => {
                self.advance();
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("u32_to_f64 produces f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::U32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::U32_TO_F64, arg })
            }
            Token::U32ToF64Norm => {
                self.advance();
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("u32_to_f64_norm produces f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::U32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::U32_TO_F64_NORM, arg })
            }
            Token::F32ToF64 => {
                self.advance();
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("f32_to_f64 produces f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::F32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F32_TO_F64, arg })
            }
            Token::F64ToF32 => {
                self.advance();
                if declared_ty != ValType::F32 {
                    return Err(ParseError { line, col, message: format!("f64_to_f32 produces f32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::F64, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F64_TO_F32, arg })
            }
            Token::I32ToU32 => {
                self.advance();
                if declared_ty != ValType::U32 {
                    return Err(ParseError { line, col, message: format!("i32_to_u32 produces u32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::I32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::I32_TO_U32, arg })
            }
            Token::U32ToI32 => {
                self.advance();
                if declared_ty != ValType::I32 {
                    return Err(ParseError { line, col, message: format!("u32_to_i32 produces i32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::U32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::U32_TO_I32, arg })
            }
            Token::I32ToF64 => {
                self.advance();
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("i32_to_f64 produces f64, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::I32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::I32_TO_F64, arg })
            }
            Token::F64ToI32 => {
                self.advance();
                if declared_ty != ValType::I32 {
                    return Err(ParseError { line, col, message: format!("f64_to_i32 produces i32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::F64, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F64_TO_I32, arg })
            }
            Token::I32ToF32 => {
                self.advance();
                if declared_ty != ValType::F32 {
                    return Err(ParseError { line, col, message: format!("i32_to_f32 produces f32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::I32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::I32_TO_F32, arg })
            }
            Token::F32ToI32 => {
                self.advance();
                if declared_ty != ValType::I32 {
                    return Err(ParseError { line, col, message: format!("f32_to_i32 produces i32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::F32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F32_TO_I32, arg })
            }
            Token::F32ToU32 => {
                self.advance();
                if declared_ty != ValType::U32 {
                    return Err(ParseError { line, col, message: format!("f32_to_u32 produces u32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::F32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::F32_TO_U32, arg })
            }
            Token::U32ToF32 => {
                self.advance();
                if declared_ty != ValType::F32 {
                    return Err(ParseError { line, col, message: format!("u32_to_f32 produces f32, got {declared_ty}") });
                }
                let arg = self.resolve_var_ident()?;
                self.check_types_match(ValType::U32, self.var_ty(arg), "arg", line, col)?;
                Ok(Inst::Conv { op: ConvOp::U32_TO_F32, arg })
            }
            Token::Hash => {
                self.advance();
                if declared_ty != ValType::U32 {
                    return Err(ParseError { line, col, message: format!("hash produces u32, got {declared_ty}") });
                }
                let lhs = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(lhs), "lhs", line, col)?;
                let rhs = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(rhs), "rhs", line, col)?;
                Ok(Inst::Binary { op: BinOp::Hash, lhs, rhs })
            }
            Token::Select => {
                self.advance();
                let cond = self.resolve_var_ident()?;
                self.check_types_match(ValType::BOOL, self.var_ty(cond), "cond", line, col)?;
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
                if declared_ty != ValType::U32 {
                    return Err(ParseError { line, col, message: format!("pack_argb produces u32, got {declared_ty}") });
                }
                let r = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(r), "r", line, col)?;
                let g = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(g), "g", line, col)?;
                let b = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(b), "b", line, col)?;
                Ok(Inst::PackArgb { r, g, b })
            }
            Token::MakeVec2 | Token::MakeVec3 | Token::MakeVec4 => {
                let (count, label) = match sp.token {
                    Token::MakeVec2 => (2u8, "make_vec2"),
                    Token::MakeVec3 => (3u8, "make_vec3"),
                    Token::MakeVec4 => (4u8, "make_vec4"),
                    _ => unreachable!(),
                };
                self.advance();
                let mut components = Vec::new();
                for _ in 0..count {
                    let v = self.resolve_var_ident()?;
                    self.check_types_match(ValType::F64, self.var_ty(v), "component", line, col)?;
                    components.push(v);
                }
                if declared_ty != (ValType::Vec { len: count, elem: ScalarType::F64 }) {
                    return Err(ParseError { line, col, message: format!("{label} result must be vec{count}<f64>, got {declared_ty}") });
                }
                Ok(Inst::MakeVec(components))
            }
            Token::ExtractX => {
                self.advance();
                let vec = self.resolve_var_ident()?;
                let vec_ty = self.var_ty(vec);
                if !matches!(vec_ty, ValType::Vec { len: 2 | 3 | 4, .. }) {
                    return Err(ParseError { line, col, message: format!("extract_x requires vec2, vec3, or vec4 argument, got {vec_ty}") });
                }
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("extract_x result must be f64, got {declared_ty}") });
                }
                Ok(Inst::VecExtract { vec, index: 0 })
            }
            Token::ExtractY => {
                self.advance();
                let vec = self.resolve_var_ident()?;
                let vec_ty = self.var_ty(vec);
                if !matches!(vec_ty, ValType::Vec { len: 2 | 3 | 4, .. }) {
                    return Err(ParseError { line, col, message: format!("extract_y requires vec2, vec3, or vec4 argument, got {vec_ty}") });
                }
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("extract_y result must be f64, got {declared_ty}") });
                }
                Ok(Inst::VecExtract { vec, index: 1 })
            }
            Token::ExtractZ => {
                self.advance();
                let vec = self.resolve_var_ident()?;
                let vec_ty = self.var_ty(vec);
                if !matches!(vec_ty, ValType::Vec { len: 3 | 4, .. }) {
                    return Err(ParseError { line, col, message: format!("extract_z requires vec3 or vec4 argument, got {vec_ty}") });
                }
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("extract_z result must be f64, got {declared_ty}") });
                }
                Ok(Inst::VecExtract { vec, index: 2 })
            }
            Token::ExtractW => {
                self.advance();
                let vec = self.resolve_var_ident()?;
                let vec_ty = self.var_ty(vec);
                if !matches!(vec_ty, ValType::Vec { len: 4, .. }) {
                    return Err(ParseError { line, col, message: format!("extract_w requires vec4 argument, got {vec_ty}") });
                }
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("extract_w result must be f64, got {declared_ty}") });
                }
                Ok(Inst::VecExtract { vec, index: 3 })
            }
            Token::VecAdd | Token::VecSub | Token::VecMul | Token::VecDiv | Token::VecMin | Token::VecMax => {
                let vec_op = match sp.token {
                    Token::VecAdd => VecBinOp::Add,
                    Token::VecSub => VecBinOp::Sub,
                    Token::VecMul => VecBinOp::Mul,
                    Token::VecDiv => VecBinOp::Div,
                    Token::VecMin => VecBinOp::Min,
                    Token::VecMax => VecBinOp::Max,
                    _ => unreachable!(),
                };
                self.advance();
                let lhs = self.resolve_var_ident()?;
                let rhs = self.resolve_var_ident()?;
                let lhs_ty = self.var_ty(lhs);
                let rhs_ty = self.var_ty(rhs);
                if lhs_ty != rhs_ty || !lhs_ty.is_vec() {
                    return Err(ParseError { line, col, message: format!("vec binary op requires matching vec types, got {lhs_ty} and {rhs_ty}") });
                }
                if declared_ty != lhs_ty {
                    return Err(ParseError { line, col, message: format!("vec binary result type must match operands: expected {lhs_ty}, got {declared_ty}") });
                }
                Ok(Inst::VecBinary { op: vec_op, lhs, rhs })
            }
            Token::VecNeg | Token::VecAbs | Token::VecNormalize => {
                let vec_op = match sp.token {
                    Token::VecNeg => VecUnaryOp::Neg,
                    Token::VecAbs => VecUnaryOp::Abs,
                    Token::VecNormalize => VecUnaryOp::Normalize,
                    _ => unreachable!(),
                };
                self.advance();
                let arg = self.resolve_var_ident()?;
                let arg_ty = self.var_ty(arg);
                if !arg_ty.is_vec() {
                    return Err(ParseError { line, col, message: format!("vec unary op requires vec type, got {arg_ty}") });
                }
                if declared_ty != arg_ty {
                    return Err(ParseError { line, col, message: format!("vec unary result type must match operand: expected {arg_ty}, got {declared_ty}") });
                }
                Ok(Inst::VecUnary { op: vec_op, arg })
            }
            Token::VecScale => {
                self.advance();
                let scalar = self.resolve_var_ident()?;
                self.check_types_match(ValType::F64, self.var_ty(scalar), "scalar", line, col)?;
                let vec = self.resolve_var_ident()?;
                let vec_ty = self.var_ty(vec);
                if !vec_ty.is_vec() {
                    return Err(ParseError { line, col, message: format!("vec_scale requires vec operand, got {vec_ty}") });
                }
                if declared_ty != vec_ty {
                    return Err(ParseError { line, col, message: format!("vec_scale result type must match vec operand: expected {vec_ty}, got {declared_ty}") });
                }
                Ok(Inst::VecScale { scalar, vec })
            }
            Token::VecDot => {
                self.advance();
                let lhs = self.resolve_var_ident()?;
                let rhs = self.resolve_var_ident()?;
                let lhs_ty = self.var_ty(lhs);
                let rhs_ty = self.var_ty(rhs);
                if lhs_ty != rhs_ty || !lhs_ty.is_vec() {
                    return Err(ParseError { line, col, message: format!("vec_dot requires matching vec types, got {lhs_ty} and {rhs_ty}") });
                }
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("vec_dot result must be f64, got {declared_ty}") });
                }
                Ok(Inst::VecDot { lhs, rhs })
            }
            Token::VecLength => {
                self.advance();
                let arg = self.resolve_var_ident()?;
                let arg_ty = self.var_ty(arg);
                if !arg_ty.is_vec() {
                    return Err(ParseError { line, col, message: format!("vec_length requires vec type, got {arg_ty}") });
                }
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("vec_length result must be f64, got {declared_ty}") });
                }
                Ok(Inst::VecLength { arg })
            }
            Token::VecCross => {
                self.advance();
                let lhs = self.resolve_var_ident()?;
                let rhs = self.resolve_var_ident()?;
                let lhs_ty = self.var_ty(lhs);
                let rhs_ty = self.var_ty(rhs);
                let vec3_f64 = ValType::Vec { len: 3, elem: ScalarType::F64 };
                if lhs_ty != vec3_f64 || rhs_ty != vec3_f64 {
                    return Err(ParseError { line, col, message: format!("vec_cross requires vec3<f64> operands, got {lhs_ty} and {rhs_ty}") });
                }
                if declared_ty != vec3_f64 {
                    return Err(ParseError { line, col, message: format!("vec_cross result must be vec3, got {declared_ty}") });
                }
                Ok(Inst::VecCross { lhs, rhs })
            }
            Token::MakeMat2 | Token::MakeMat3 | Token::MakeMat4 => {
                let (count, label) = match sp.token {
                    Token::MakeMat2 => (2u8, "make_mat2"),
                    Token::MakeMat3 => (3u8, "make_mat3"),
                    Token::MakeMat4 => (4u8, "make_mat4"),
                    _ => unreachable!(),
                };
                self.advance();
                let elem = match declared_ty {
                    ValType::Mat { size, elem } if size == count => elem,
                    _ => return Err(ParseError { line, col, message: format!("{label} result must be mat{count}<_>, got {declared_ty}") }),
                };
                let col_ty = ValType::Vec { len: count, elem };
                let mut cols = Vec::new();
                for _ in 0..count {
                    let v = self.resolve_var_ident()?;
                    self.check_types_match(col_ty, self.var_ty(v), "column", line, col)?;
                    cols.push(v);
                }
                Ok(Inst::MakeMat(cols))
            }
            Token::MatMulVec => {
                self.advance();
                let mat = self.resolve_var_ident()?;
                let vec = self.resolve_var_ident()?;
                let mat_ty = self.var_ty(mat);
                let vec_ty = self.var_ty(vec);
                let (size, elem) = match mat_ty {
                    ValType::Mat { size, elem } => (size, elem),
                    _ => return Err(ParseError { line, col, message: format!("mat_mul_vec requires mat operand, got {mat_ty}") }),
                };
                let expected_vec = ValType::Vec { len: size, elem };
                if vec_ty != expected_vec {
                    return Err(ParseError { line, col, message: format!("mat_mul_vec vec operand must be {expected_vec}, got {vec_ty}") });
                }
                if declared_ty != expected_vec {
                    return Err(ParseError { line, col, message: format!("mat_mul_vec result must be {expected_vec}, got {declared_ty}") });
                }
                Ok(Inst::MatMulVec { mat, vec })
            }
            Token::MatMul => {
                self.advance();
                let lhs = self.resolve_var_ident()?;
                let rhs = self.resolve_var_ident()?;
                let lhs_ty = self.var_ty(lhs);
                let rhs_ty = self.var_ty(rhs);
                if !lhs_ty.is_mat() || lhs_ty != rhs_ty {
                    return Err(ParseError { line, col, message: format!("mat_mul requires matching mat types, got {lhs_ty} and {rhs_ty}") });
                }
                if declared_ty != lhs_ty {
                    return Err(ParseError { line, col, message: format!("mat_mul result type must match operands: expected {lhs_ty}, got {declared_ty}") });
                }
                Ok(Inst::MatMul { lhs, rhs })
            }
            Token::MatTranspose => {
                self.advance();
                let arg = self.resolve_var_ident()?;
                let arg_ty = self.var_ty(arg);
                if !arg_ty.is_mat() {
                    return Err(ParseError { line, col, message: format!("mat_transpose requires mat type, got {arg_ty}") });
                }
                if declared_ty != arg_ty {
                    return Err(ParseError { line, col, message: format!("mat_transpose result type must match operand: expected {arg_ty}, got {declared_ty}") });
                }
                Ok(Inst::MatTranspose { arg })
            }
            Token::MatCol0 | Token::MatCol1 | Token::MatCol2 | Token::MatCol3 => {
                let index: u8 = match sp.token {
                    Token::MatCol0 => 0,
                    Token::MatCol1 => 1,
                    Token::MatCol2 => 2,
                    Token::MatCol3 => 3,
                    _ => unreachable!(),
                };
                self.advance();
                let mat = self.resolve_var_ident()?;
                let mat_ty = self.var_ty(mat);
                let (size, elem) = match mat_ty {
                    ValType::Mat { size, elem } => (size, elem),
                    _ => return Err(ParseError { line, col, message: format!("mat_col{index} requires mat type, got {mat_ty}") }),
                };
                if index >= size {
                    return Err(ParseError { line, col, message: format!("mat_col{index} index out of bounds for mat{size}") });
                }
                let expected = ValType::Vec { len: size, elem };
                if declared_ty != expected {
                    return Err(ParseError { line, col, message: format!("mat_col{index} result must be {expected}, got {declared_ty}") });
                }
                Ok(Inst::MatCol { mat, index })
            }
            Token::BufLoad => {
                self.advance();
                if declared_ty != ValType::F64 {
                    return Err(ParseError { line, col, message: format!("buf_load produces f64, got {declared_ty}") });
                }
                let (buf_name, bline, bcol) = self.expect_ident()?;
                let buf = *self.buf_names.get(&buf_name).ok_or_else(|| ParseError {
                    line: bline, col: bcol, message: format!("unknown buffer '{buf_name}'"),
                })?;
                let x = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(x), "buf_load x", line, col)?;
                let y = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(y), "buf_load y", line, col)?;
                Ok(Inst::BufLoad { buf, x, y })
            }
            Token::BufStore => {
                self.advance();
                // buf_store doesn't produce a value, but we still need a binding.
                // The declared type should be u32 (dummy, value 0).
                let (buf_name, bline, bcol) = self.expect_ident()?;
                let buf = *self.buf_names.get(&buf_name).ok_or_else(|| ParseError {
                    line: bline, col: bcol, message: format!("unknown buffer '{buf_name}'"),
                })?;
                if !self.buf_decls[buf as usize].is_output {
                    return Err(ParseError { line: bline, col: bcol, message: format!("buffer '{buf_name}' is read-only") });
                }
                let x = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(x), "buf_store x", line, col)?;
                let y = self.parse_operand(ValType::U32)?;
                self.check_types_match(ValType::U32, self.var_ty(y), "buf_store y", line, col)?;
                let val = self.parse_operand(ValType::F64)?;
                self.check_types_match(ValType::F64, self.var_ty(val), "buf_store val", line, col)?;
                Ok(Inst::BufStore { buf, x, y, val })
            }
            Token::Ident(name) if self.inline_defs.contains_key(name) => {
                let func_name = name.clone();
                self.advance();
                self.expand_inline(&func_name, declared_ty, line, col)
            }
            _ => Err(ParseError {
                line: sp.line,
                col: sp.col,
                message: format!("expected instruction, got {:?}", sp.token),
            }),
        }
    }

    fn parse_const_literal(&mut self, declared_ty: ValType) -> Result<Inst, ParseError> {
        let sp = self.peek().clone();
        match (&sp.token, declared_ty) {
            (Token::FloatLit(v), ValType::Scalar(ScalarType::F64)) => {
                let v = *v;
                self.advance();
                Ok(Inst::Const(Const::F64(v)))
            }
            (Token::FloatLit(v), ValType::Scalar(ScalarType::F32)) => {
                let v = *v as f32;
                self.advance();
                // Consume optional f32 suffix
                if self.peek().token == Token::TyF32 {
                    self.advance();
                }
                Ok(Inst::Const(Const::F32(v)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I8)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI8 { self.advance(); }
                if v > i8::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i8 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I8(v as i8)))
            }
            (Token::FloatLit(v), ValType::Scalar(ScalarType::I8)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI8 { self.advance(); }
                if v < i8::MIN as f64 || v > i8::MAX as f64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i8 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I8(v as i8)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U8)) => {
                let v = *v;
                self.advance();
                if v > u8::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("u8 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::U8(v as u8)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I16)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI16 { self.advance(); }
                if v > i16::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i16 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I16(v as i16)))
            }
            (Token::FloatLit(v), ValType::Scalar(ScalarType::I16)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI16 { self.advance(); }
                if v < i16::MIN as f64 || v > i16::MAX as f64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i16 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I16(v as i16)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U16)) => {
                let v = *v;
                self.advance();
                if v > u16::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("u16 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::U16(v as u16)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U32)) => {
                let v = *v;
                self.advance();
                if v > u32::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("u32 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::U32(v as u32)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I32)) => {
                let v = *v;
                self.advance();
                // Consume optional i32 suffix
                if self.peek().token == Token::TyI32 {
                    self.advance();
                }
                if v > i32::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i32 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I32(v as i32)))
            }
            // Negative integers come through as FloatLit from the lexer
            (Token::FloatLit(v), ValType::Scalar(ScalarType::I32)) => {
                let v = *v;
                self.advance();
                // Consume optional i32 suffix
                if self.peek().token == Token::TyI32 {
                    self.advance();
                }
                if v < i32::MIN as f64 || v > i32::MAX as f64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i32 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I32(v as i32)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::I64)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI64 { self.advance(); }
                if v > i64::MAX as u64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i64 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I64(v as i64)))
            }
            (Token::FloatLit(v), ValType::Scalar(ScalarType::I64)) => {
                let v = *v;
                self.advance();
                if self.peek().token == Token::TyI64 { self.advance(); }
                if v < i64::MIN as f64 || v > i64::MAX as f64 {
                    return Err(ParseError { line: sp.line, col: sp.col, message: format!("i64 literal out of range: {v}") });
                }
                Ok(Inst::Const(Const::I64(v as i64)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::U64)) => {
                let v = *v;
                self.advance();
                Ok(Inst::Const(Const::U64(v)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::F64)) => {
                let v = *v as f64;
                self.advance();
                Ok(Inst::Const(Const::F64(v)))
            }
            (Token::IntLit(v), ValType::Scalar(ScalarType::F32)) => {
                let v = *v as f32;
                self.advance();
                // Consume optional f32 suffix
                if self.peek().token == Token::TyF32 {
                    self.advance();
                }
                Ok(Inst::Const(Const::F32(v)))
            }
            (Token::True, ValType::Scalar(ScalarType::Bool)) => {
                self.advance();
                Ok(Inst::Const(Const::Bool(true)))
            }
            (Token::False, ValType::Scalar(ScalarType::Bool)) => {
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
    /// Returns (statement, implicit_stmts, expanded_body_items).
    /// expanded_body_items contains While blocks from inline expansion.
    fn parse_statement(&mut self) -> Result<(Statement, Vec<Statement>, Vec<BodyItem>), ParseError> {
        let (var_name, line, col) = self.expect_ident()?;
        self.expect_tok(&Token::Colon)?;
        let ty = self.parse_type()?;
        self.expect_tok(&Token::Equals)?;

        self.implicit_stmts.clear();
        self.expanded_body_items.clear();
        let inst = self.parse_instruction(ty, line, col)?;

        let implicits = std::mem::take(&mut self.implicit_stmts);
        let expanded = std::mem::take(&mut self.expanded_body_items);
        let var = self.alloc_var(var_name.clone(), ty, line, col)?;
        let stmt = Statement {
            binding: Binding { var, name: var_name, ty },
            inst,
        };
        Ok((stmt, implicits, expanded))
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

        // Parse cond_body: statements/whiles until `cond`
        let mut cond_body: Vec<BodyItem> = Vec::new();
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
                Token::While => {
                    let w = self.parse_while()?;
                    cond_body.push(BodyItem::While(w));
                }
                _ => {
                    let (stmt, implicits, expanded) = self.parse_statement()?;
                    for imp in implicits {
                        cond_body.push(BodyItem::Stmt(imp));
                    }
                    cond_body.extend(expanded);
                    cond_body.push(BodyItem::Stmt(stmt));
                }
            }
        }

        // Parse `cond <var>`
        self.expect_tok(&Token::Cond)?;
        let (cond_name, cond_line, cond_col) = self.expect_ident()?;
        let cond = self.resolve_var(&cond_name, cond_line, cond_col)?;
        self.check_types_match(ValType::BOOL, self.var_ty(cond), "cond", cond_line, cond_col)?;

        // Parse body: statements/whiles until `yield`
        let mut body: Vec<BodyItem> = Vec::new();
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
                Token::While => {
                    let w = self.parse_while()?;
                    body.push(BodyItem::While(w));
                }
                _ => {
                    let (stmt, implicits, expanded) = self.parse_statement()?;
                    for imp in implicits {
                        body.push(BodyItem::Stmt(imp));
                    }
                    body.extend(expanded);
                    body.push(BodyItem::Stmt(stmt));
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
        let carry_entries: Vec<(String, Var, ValType)> = carry.iter()
            .map(|cv| (cv.binding.name.clone(), cv.binding.var, cv.binding.ty))
            .collect();
        self.vars = outer_vars;
        for (name, var, ty) in carry_entries {
            self.vars.insert(name, var);
            self.var_types.insert(var, ty);
        }

        Ok(While { carry, cond_body, cond, body, yields })
    }

    /// Parse an inline function definition into `self.inline_defs`.
    /// Uses a temporary isolated scope for the function body.
    fn parse_inline_def(&mut self) -> Result<(), ParseError> {
        self.expect_tok(&Token::Inline)?;
        let (name, def_line, def_col) = self.expect_ident()?;
        if self.inline_defs.contains_key(&name) {
            return Err(ParseError {
                line: def_line,
                col: def_col,
                message: format!("duplicate inline function: '{name}'"),
            });
        }

        // Save outer parser state
        let saved_vars = std::mem::take(&mut self.vars);
        let saved_var_types = std::mem::take(&mut self.var_types);
        let saved_next_var = self.next_var;
        self.next_var = 0;

        // Parse parameters into isolated scope
        let raw_params = self.parse_params()?;
        let params: Vec<(String, Var, ValType)> = raw_params
            .iter()
            .map(|b| (b.name.clone(), b.var, b.ty))
            .collect();

        self.expect_tok(&Token::Arrow)?;
        let return_ty = self.parse_type()?;
        self.expect_tok(&Token::LBrace)?;

        // Parse body items until `return`
        let mut body = Vec::new();
        loop {
            let sp = self.peek().clone();
            match &sp.token {
                Token::Return => break,
                Token::RBrace | Token::Eof => {
                    return Err(ParseError {
                        line: sp.line,
                        col: sp.col,
                        message: "expected 'return' in inline function body".to_string(),
                    });
                }
                Token::While => {
                    let w = self.parse_while()?;
                    let carry_names = w.carry.iter().map(|cv| cv.binding.name.clone()).collect();
                    let cond_body_names = collect_body_item_names(&w.cond_body);
                    let body_names = collect_body_item_names(&w.body);
                    body.push(InlineBodyItem::While { w, carry_names, cond_body_names, body_names });
                }
                _ => {
                    let (stmt, implicits, _expanded) = self.parse_statement()?;
                    for imp in implicits {
                        body.push(InlineBodyItem::Stmt {
                            name: imp.binding.name.clone(),
                            ty: imp.binding.ty,
                            var: imp.binding.var,
                            inst: imp.inst,
                        });
                    }
                    body.push(InlineBodyItem::Stmt {
                        name: stmt.binding.name.clone(),
                        ty: stmt.binding.ty,
                        var: stmt.binding.var,
                        inst: stmt.inst,
                    });
                }
            }
        }

        // Parse `return <var>`
        self.expect_tok(&Token::Return)?;
        let (ret_name, ret_line, ret_col) = self.expect_ident()?;
        let ret_var = self.resolve_var(&ret_name, ret_line, ret_col)?;
        let ret_var_ty = self.var_ty(ret_var);
        if ret_var_ty != return_ty {
            return Err(ParseError {
                line: ret_line,
                col: ret_col,
                message: format!(
                    "return type mismatch: inline '{name}' returns {return_ty}, but '{ret_name}' is {ret_var_ty}"
                ),
            });
        }

        // Find the instruction that produced ret_var — it becomes the call-site instruction.
        // Remove it from body so it doesn't get expanded as a separate stmt.
        let ret_idx = body.iter().rposition(|item| match item {
            InlineBodyItem::Stmt { var, .. } => *var == ret_var,
            _ => false,
        });
        let return_inst = if let Some(idx) = ret_idx {
            match body.remove(idx) {
                InlineBodyItem::Stmt { inst, .. } => inst,
                _ => unreachable!(),
            }
        } else {
            // Return var is a while carry var or similar — create an identity via add 0 for f64,
            // or a select(true, v, v) as a generic copy. Actually, we'll just store the var
            // and during expansion, map the call-site var to whatever the return var maps to.
            // We use a sentinel and handle it specially in expand_inline.
            // For now, store a Const(F64(0.0)) as a placeholder — expand_inline will detect
            // return_var != None and use the remapped var directly.
            Inst::Const(Const::F64(0.0)) // placeholder, won't be used
        };
        let return_var_is_stmt = ret_idx.is_some();

        self.expect_tok(&Token::RBrace)?;

        // Restore outer parser state
        self.vars = saved_vars;
        self.var_types = saved_var_types;
        self.next_var = saved_next_var;

        self.inline_defs.insert(name, InlineDef {
            params,
            return_ty,
            return_inst,
            return_var: ret_var,
            return_var_is_stmt: return_var_is_stmt,
            body,
        });
        Ok(())
    }

    /// Expand an inline function call at the current call site.
    /// Parses arguments, creates prefixed variables, appends expanded body to `self.implicit_stmts`.
    /// Returns the remapped return instruction.
    fn expand_inline(
        &mut self,
        func_name: &str,
        declared_ty: ValType,
        line: usize,
        col: usize,
    ) -> Result<Inst, ParseError> {
        let def = self.inline_defs[func_name].clone();
        let call_id = self.inline_call_count;
        self.inline_call_count += 1;

        // Type-check return type
        if def.return_ty != declared_ty {
            return Err(ParseError {
                line,
                col,
                message: format!(
                    "type mismatch: inline '{func_name}' returns {}, but declared type is {declared_ty}",
                    def.return_ty
                ),
            });
        }

        // Parse arguments and build var remap: inline param Var -> caller Var
        let mut var_map: HashMap<Var, Var> = HashMap::new();
        for (param_name, param_var, param_ty) in &def.params {
            let arg = self.parse_operand(*param_ty)?;
            let arg_ty = self.var_ty(arg);
            if arg_ty != *param_ty {
                return Err(ParseError {
                    line,
                    col,
                    message: format!(
                        "type mismatch for argument '{param_name}' of inline '{func_name}': expected {param_ty}, got {arg_ty}"
                    ),
                });
            }
            var_map.insert(*param_var, arg);
        }

        let prefix = format!("__{func_name}_{call_id}_");

        // Expand body items
        for item in &def.body {
            match item {
                InlineBodyItem::Stmt { name, ty, var, inst } => {
                    let new_name = format!("{prefix}{name}");
                    let new_var = Var(self.next_var);
                    self.next_var += 1;
                    self.vars.insert(new_name.clone(), new_var);
                    self.var_types.insert(new_var, *ty);
                    var_map.insert(*var, new_var);

                    let new_inst = Self::remap_inst(inst, &var_map);
                    self.implicit_stmts.push(Statement {
                        binding: Binding { var: new_var, name: new_name, ty: *ty },
                        inst: new_inst,
                    });
                }
                InlineBodyItem::While { w, carry_names, cond_body_names, body_names } => {
                    let new_while = self.remap_while(w, &mut var_map, &prefix, carry_names, cond_body_names, body_names);
                    self.expanded_body_items.push(BodyItem::While(new_while));
                }
            }
        }

        // Remap the return instruction
        if def.return_var_is_stmt {
            Ok(Self::remap_inst(&def.return_inst, &var_map))
        } else {
            // Return var is a carry var — create an identity op to copy its value
            let remapped = var_map[&def.return_var];
            match def.return_ty {
                ValType::Scalar(ScalarType::F32) => {
                    let zero = self.alloc_implicit(ValType::F32, Const::F32(0.0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::F64) => {
                    let zero = self.alloc_implicit(ValType::F64, Const::F64(0.0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::I8) => {
                    let zero = self.alloc_implicit(ValType::I8, Const::I8(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::U8) => {
                    let zero = self.alloc_implicit(ValType::U8, Const::U8(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::I16) => {
                    let zero = self.alloc_implicit(ValType::I16, Const::I16(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::U16) => {
                    let zero = self.alloc_implicit(ValType::U16, Const::U16(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::I32) => {
                    let zero = self.alloc_implicit(ValType::I32, Const::I32(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::U32) => {
                    let zero = self.alloc_implicit(ValType::U32, Const::U32(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::I64) => {
                    let zero = self.alloc_implicit(ValType::I64, Const::I64(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::U64) => {
                    let zero = self.alloc_implicit(ValType::U64, Const::U64(0));
                    Ok(Inst::Binary { op: BinOp::Add, lhs: remapped, rhs: zero })
                }
                ValType::Scalar(ScalarType::Bool) => {
                    let f = self.alloc_implicit(ValType::BOOL, Const::Bool(false));
                    Ok(Inst::Binary { op: BinOp::Or, lhs: remapped, rhs: f })
                }
                ValType::Vec { .. } => {
                    // Identity: vec + zero-vec. Use vec_add with a zero-scaled copy.
                    let zero = self.alloc_implicit(ValType::F64, Const::F64(0.0));
                    let zero_vec_name = format!("__identity_zero_vec_{}", self.next_var);
                    let zero_vec = Var(self.next_var);
                    self.next_var += 1;
                    self.vars.insert(zero_vec_name.clone(), zero_vec);
                    self.var_types.insert(zero_vec, def.return_ty);
                    self.implicit_stmts.push(Statement {
                        binding: Binding { var: zero_vec, name: zero_vec_name, ty: def.return_ty },
                        inst: Inst::VecScale { scalar: zero, vec: remapped },
                    });
                    Ok(Inst::VecBinary { op: VecBinOp::Add, lhs: remapped, rhs: zero_vec })
                }
                ValType::Mat { .. } => {
                    // Identity for matrix: multiply by identity matrix or just return.
                    // For now, use MatMul with self (a * I = a), but we don't have identity
                    // matrix construction yet. Use a transpose-of-transpose as identity.
                    let transposed_name = format!("__identity_tmp_{}", self.next_var);
                    let transposed = Var(self.next_var);
                    self.next_var += 1;
                    self.vars.insert(transposed_name.clone(), transposed);
                    self.var_types.insert(transposed, def.return_ty);
                    self.implicit_stmts.push(Statement {
                        binding: Binding { var: transposed, name: transposed_name, ty: def.return_ty },
                        inst: Inst::MatTranspose { arg: remapped },
                    });
                    Ok(Inst::MatTranspose { arg: transposed })
                }
            }
        }
    }

    fn remap_inst(inst: &Inst, var_map: &HashMap<Var, Var>) -> Inst {
        let remap = |v: &Var| -> Var { var_map.get(v).copied().unwrap_or(*v) };
        match inst {
            Inst::Const(c) => Inst::Const(*c),
            Inst::Binary { op, lhs, rhs } => Inst::Binary { op: *op, lhs: remap(lhs), rhs: remap(rhs) },
            Inst::Unary { op, arg } => Inst::Unary { op: *op, arg: remap(arg) },
            Inst::Cmp { op, lhs, rhs } => Inst::Cmp { op: *op, lhs: remap(lhs), rhs: remap(rhs) },
            Inst::Conv { op, arg } => Inst::Conv { op: *op, arg: remap(arg) },
            Inst::Select { cond, then_val, else_val } => Inst::Select {
                cond: remap(cond),
                then_val: remap(then_val),
                else_val: remap(else_val),
            },
            Inst::PackArgb { r, g, b } => Inst::PackArgb { r: remap(r), g: remap(g), b: remap(b) },
            Inst::MakeVec(components) => Inst::MakeVec(components.iter().map(&remap).collect()),
            Inst::VecExtract { vec, index } => Inst::VecExtract { vec: remap(vec), index: *index },
            Inst::VecBinary { op, lhs, rhs } => Inst::VecBinary { op: *op, lhs: remap(lhs), rhs: remap(rhs) },
            Inst::VecScale { scalar, vec } => Inst::VecScale { scalar: remap(scalar), vec: remap(vec) },
            Inst::VecUnary { op, arg } => Inst::VecUnary { op: *op, arg: remap(arg) },
            Inst::VecDot { lhs, rhs } => Inst::VecDot { lhs: remap(lhs), rhs: remap(rhs) },
            Inst::VecLength { arg } => Inst::VecLength { arg: remap(arg) },
            Inst::VecCross { lhs, rhs } => Inst::VecCross { lhs: remap(lhs), rhs: remap(rhs) },
            Inst::MakeMat(cols) => Inst::MakeMat(cols.iter().map(&remap).collect()),
            Inst::MatMulVec { mat, vec } => Inst::MatMulVec { mat: remap(mat), vec: remap(vec) },
            Inst::MatMul { lhs, rhs } => Inst::MatMul { lhs: remap(lhs), rhs: remap(rhs) },
            Inst::MatTranspose { arg } => Inst::MatTranspose { arg: remap(arg) },
            Inst::MatCol { mat, index } => Inst::MatCol { mat: remap(mat), index: *index },
            Inst::BufLoad { buf, x, y } => Inst::BufLoad { buf: *buf, x: remap(x), y: remap(y) },
            Inst::BufStore { buf, x, y, val } => Inst::BufStore { buf: *buf, x: remap(x), y: remap(y), val: remap(val) },
        }
    }

    /// Remap a While struct, allocating fresh prefixed Vars for carry vars and inner body items.
    fn remap_while(
        &mut self,
        w: &While,
        var_map: &mut HashMap<Var, Var>,
        prefix: &str,
        carry_names: &[String],
        cond_body_names: &[String],
        body_names: &[String],
    ) -> While {
        // Remap carry vars
        let mut new_carry = Vec::new();
        for (cv, orig_name) in w.carry.iter().zip(carry_names) {
            let new_name = format!("{prefix}{orig_name}");
            let new_var = Var(self.next_var);
            self.next_var += 1;
            self.vars.insert(new_name.clone(), new_var);
            self.var_types.insert(new_var, cv.binding.ty);
            var_map.insert(cv.binding.var, new_var);
            new_carry.push(CarryVar {
                binding: Binding { var: new_var, name: new_name, ty: cv.binding.ty },
                init: var_map.get(&cv.init).copied().unwrap_or(cv.init),
            });
        }

        // Remap cond_body
        let mut name_iter = cond_body_names.iter();
        let new_cond_body = self.remap_body_items(&w.cond_body, var_map, prefix, &mut name_iter);

        let new_cond = var_map.get(&w.cond).copied().unwrap_or(w.cond);

        // Remap body
        let mut name_iter = body_names.iter();
        let new_body = self.remap_body_items(&w.body, var_map, prefix, &mut name_iter);

        let new_yields: Vec<Var> = w.yields.iter()
            .map(|v| var_map.get(v).copied().unwrap_or(*v))
            .collect();

        While {
            carry: new_carry,
            cond_body: new_cond_body,
            cond: new_cond,
            body: new_body,
            yields: new_yields,
        }
    }

    /// Recursively remap body items, consuming names from the iterator.
    fn remap_body_items<'a>(
        &mut self,
        items: &[BodyItem],
        var_map: &mut HashMap<Var, Var>,
        prefix: &str,
        names: &mut impl Iterator<Item = &'a String>,
    ) -> Vec<BodyItem> {
        let mut result = Vec::new();
        for item in items {
            match item {
                BodyItem::Stmt(stmt) => {
                    let orig_name = names.next().expect("name iterator exhausted");
                    let new_name = format!("{prefix}{orig_name}");
                    let new_var = Var(self.next_var);
                    self.next_var += 1;
                    self.vars.insert(new_name.clone(), new_var);
                    self.var_types.insert(new_var, stmt.binding.ty);
                    var_map.insert(stmt.binding.var, new_var);
                    result.push(BodyItem::Stmt(Statement {
                        binding: Binding { var: new_var, name: new_name, ty: stmt.binding.ty },
                        inst: Self::remap_inst(&stmt.inst, var_map),
                    }));
                }
                BodyItem::While(inner_w) => {
                    // Consume carry names, cond_body names, body names from the flat iterator
                    let carry_names: Vec<String> = inner_w.carry.iter()
                        .map(|_| names.next().expect("name iterator exhausted").clone())
                        .collect();
                    // Count body item names needed
                    let cond_count = count_body_item_names(&inner_w.cond_body);
                    let cond_names: Vec<String> = (0..cond_count)
                        .map(|_| names.next().expect("name iterator exhausted").clone())
                        .collect();
                    let body_count = count_body_item_names(&inner_w.body);
                    let body_names: Vec<String> = (0..body_count)
                        .map(|_| names.next().expect("name iterator exhausted").clone())
                        .collect();
                    let remapped = self.remap_while(inner_w, var_map, prefix, &carry_names, &cond_names, &body_names);
                    result.push(BodyItem::While(remapped));
                }
            }
        }
        result
    }

    fn parse_params(&mut self) -> Result<Vec<Binding>, ParseError> {
        self.expect_tok(&Token::LParen)?;
        let mut params = Vec::new();
        loop {
            let sp = self.peek().clone();
            if sp.token == Token::RParen {
                self.advance();
                break;
            }
            if !params.is_empty() {
                self.expect_tok(&Token::Comma)?;
            }
            let (name, line, col) = self.expect_ident()?;
            self.expect_tok(&Token::Colon)?;
            let ty = self.parse_type()?;
            let var = self.alloc_var(name.clone(), ty, line, col)?;
            params.push(Binding { var, name, ty });
        }
        Ok(params)
    }

    fn parse_kernel(&mut self) -> Result<Kernel, ParseError> {
        self.expect_tok(&Token::Kernel)?;
        let (name, _line, _col) = self.expect_ident()?;
        let params = self.parse_params()?;
        self.expect_tok(&Token::Arrow)?;
        let return_ty = self.parse_type()?;

        // Optional buffer declarations
        if self.peek().token == Token::Buffers {
            self.advance();
            self.expect_tok(&Token::LParen)?;
            loop {
                let sp = self.peek().clone();
                if sp.token == Token::RParen {
                    self.advance();
                    break;
                }
                if !self.buf_decls.is_empty() {
                    self.expect_tok(&Token::Comma)?;
                }
                let (buf_name, _bl, _bc) = self.expect_ident()?;
                self.expect_tok(&Token::Colon)?;
                let mode_sp = self.peek().clone();
                let is_output = match &mode_sp.token {
                    Token::Read_ => { self.advance(); false }
                    Token::Write_ => { self.advance(); true }
                    _ => return Err(ParseError {
                        line: mode_sp.line, col: mode_sp.col,
                        message: "expected 'read' or 'write'".to_string(),
                    }),
                };
                let idx = self.buf_decls.len() as u32;
                self.buf_names.insert(buf_name.clone(), idx);
                self.buf_decls.push(BufDecl { name: buf_name, is_output });
            }
        }

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
                    let (stmt, implicits, expanded) = self.parse_statement()?;
                    for imp in implicits {
                        body.push(BodyItem::Stmt(imp));
                    }
                    body.extend(expanded);
                    body.push(BodyItem::Stmt(stmt));
                }
            }
        }

        self.expect_tok(&Token::Emit)?;
        let (emit_name, emit_line, emit_col) = self.expect_ident()?;
        let emit_var = self.resolve_var(&emit_name, emit_line, emit_col)?;
        let emit_ty = self.var_ty(emit_var);
        if emit_ty != return_ty {
            return Err(ParseError {
                line: emit_line,
                col: emit_col,
                message: format!("emit type mismatch: kernel returns {return_ty}, but '{emit_name}' is {emit_ty}"),
            });
        }

        self.expect_tok(&Token::RBrace)?;

        let buffers = std::mem::take(&mut self.buf_decls);
        Ok(Kernel { name, params, return_ty, body, emit: emit_var, buffers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gradient() {
        let src = r#"
kernel gradient(x: f64, y: f64) -> u32 {
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
        assert_eq!(kernel.params.len(), 2);
        assert_eq!(kernel.params[0].name, "x");
        assert_eq!(kernel.params[1].name, "y");
        // x=Var(0), y=Var(1), implicit 255.0=Var(2), r=Var(3),
        // r_u=Var(4), implicit 255.0=Var(5), g=Var(6), g_u=Var(7),
        // b=Var(8), pixel=Var(9)
        assert_eq!(kernel.emit, Var(9));
    }

    #[test]
    fn test_parse_solid_color() {
        let src = r#"
kernel solid(x: f64, y: f64) -> u32 {
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
kernel bad(x: f64, y: f64) -> u32 {
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
kernel bad(x: f64, y: f64) -> u32 {
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
kernel bad(x: f64, y: f64) -> u32 {
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
        let src = r#"kernel gradient(x: f64, y: f64) -> u32 {
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
kernel loop_test(x: f64, y: f64) -> u32 {
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
kernel carry_test(x: f64, y: f64) -> u32 {
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
        let src = include_str!("../../examples/basic/mandelbrot/mandelbrot.pdir");
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "mandelbrot");
        // Verify roundtrip
        let printed = crate::lang::printer::print(&kernel);
        let kernel2 = parse(&printed).unwrap();
        assert_eq!(kernel2.name, kernel.name);
    }

    #[test]
    fn test_parse_sdf_flower() {
        let src = include_str!("../../examples/sdf/sdf_flower/sdf_flower.pdir");
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "sdf_flower");
        // Verify roundtrip: expanded flat form should re-parse successfully
        let printed = crate::lang::printer::print(&kernel);
        let kernel2 = parse(&printed).unwrap();
        assert_eq!(kernel2.name, kernel.name);
        // Body length may differ slightly due to inline expansion intermediates
        // but the re-parsed kernel should be valid
    }

    #[test]
    fn test_parse_while_inner_vars_not_visible_outside() {
        // Variables defined inside the while body should not be visible after it
        let src = r#"
kernel scope_test(x: f64, y: f64) -> u32 {
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

    // ── Inline function tests ──────────────────────────────────────

    #[test]
    fn test_inline_basic_expansion() {
        let src = r#"
inline double(a: f64) -> f64 {
    result: f64 = add a a
    return result
}

kernel test_inline(x: f64, y: f64) -> u32 {
    d: f64 = double x
    r: u32 = f64_to_u32 d
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        assert_eq!(kernel.name, "test_inline");
        // d should be computed as add x x (expanded)
        // Body should have implicit __double_0_* stmts filtered by the stmt being `d`
        // Actually the return inst becomes d's instruction, so no separate stmt for result.
        // Let's verify via roundtrip that the expansion works.
        let printed = crate::lang::printer::print(&kernel);
        // The expanded form should show `d: f64 = add x x`
        assert!(printed.contains("d: f64 = add x x"), "got:\n{printed}");
    }

    #[test]
    fn test_inline_multiple_calls() {
        let src = r#"
inline double(a: f64) -> f64 {
    result: f64 = add a a
    return result
}

kernel test_inline2(x: f64, y: f64) -> u32 {
    dx: f64 = double x
    dy: f64 = double y
    sum: f64 = add dx dy
    r: u32 = f64_to_u32 sum
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        let printed = crate::lang::printer::print(&kernel);
        assert!(printed.contains("dx: f64 = add x x"), "got:\n{printed}");
        assert!(printed.contains("dy: f64 = add y y"), "got:\n{printed}");
    }

    #[test]
    fn test_inline_with_body_stmts() {
        let src = r#"
inline smin(a: f64, b: f64, k: f64) -> f64 {
    ka: f64 = mul k a
    neg_ka: f64 = neg ka
    kb: f64 = mul k b
    neg_kb: f64 = neg kb
    ea: f64 = exp neg_ka
    eb: f64 = exp neg_kb
    esum: f64 = add ea eb
    log_sum: f64 = log esum
    neg_log: f64 = neg log_sum
    result: f64 = div neg_log k
    return result
}

kernel test_smin(x: f64, y: f64) -> u32 {
    d: f64 = smin x y 12.0
    r: u32 = f64_to_u32 d
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        // Should have expanded body items with __smin_0_ prefix
        let has_prefixed = kernel.body.iter().any(|item| {
            if let BodyItem::Stmt(s) = item {
                s.binding.name.starts_with("__smin_0_")
            } else {
                false
            }
        });
        assert!(has_prefixed, "expected __smin_0_ prefixed statements in expanded body");
    }

    #[test]
    fn test_inline_type_mismatch_return() {
        let src = r#"
inline bad(a: f64) -> f64 {
    result: f64 = add a a
    return result
}

kernel test_bad(x: f64, y: f64) -> u32 {
    d: u32 = bad x
    pixel: u32 = pack_argb d d d
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("type mismatch"), "got: {}", err.message);
    }

    #[test]
    fn test_inline_type_mismatch_arg() {
        let src = r#"
inline double(a: f64) -> f64 {
    result: f64 = add a a
    return result
}

kernel test_bad(x: f64, y: f64) -> u32 {
    i: u32 = const 1
    d: f64 = double i
    r: u32 = f64_to_u32 d
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("type mismatch"), "got: {}", err.message);
    }

    #[test]
    fn test_inline_scope_isolation() {
        // Variables from outer scope should not be accessible inside inline body
        let src = r#"
inline bad(a: f64) -> f64 {
    result: f64 = add a outer
    return result
}

kernel test_scope(x: f64, y: f64) -> u32 {
    outer: f64 = const 1.0
    d: f64 = bad x
    r: u32 = f64_to_u32 d
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let err = parse(src).unwrap_err();
        assert!(err.message.contains("undefined variable"), "got: {}", err.message);
    }

    #[test]
    fn test_inline_with_while() {
        let src = r#"
inline iterate(start: f64, limit: f64) -> f64 {
    while carry(val: f64 = start) {
        done: bool = ge val limit
        cont: bool = not done
        cond cont
        next: f64 = add val 1.0
        yield next
    }
    return val
}

kernel test_while_inline(x: f64, y: f64) -> u32 {
    limit: f64 = const 10.0
    result: f64 = iterate x limit
    r: u32 = f64_to_u32 result
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        // Should have a While body item from expansion
        let has_while = kernel.body.iter().any(|item| matches!(item, BodyItem::While(_)));
        assert!(has_while, "expected expanded while in kernel body");
    }

    #[test]
    fn test_inline_with_literal_args() {
        let src = r#"
inline scale(a: f64, factor: f64) -> f64 {
    result: f64 = mul a factor
    return result
}

kernel test_lit(x: f64, y: f64) -> u32 {
    d: f64 = scale x 2.5
    r: u32 = f64_to_u32 d
    pixel: u32 = pack_argb r r r
    emit pixel
}
"#;
        let kernel = parse(src).unwrap();
        let printed = crate::lang::printer::print(&kernel);
        assert!(printed.contains("d: f64 = mul x 2.5"), "got:\n{printed}");
    }
}
