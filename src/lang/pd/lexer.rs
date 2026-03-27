use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Use,
    Kernel,
    Fn,
    Let,
    Emit,
    Return,
    While,
    BreakIf,
    Yield,
    If,
    Else,
    True,
    False,
    Buffers,
    Textures,
    Read,
    Write,
    Struct,
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
    // Literals
    F32Lit(f32),
    FloatLit(f64),
    I8Lit(i8),
    U8Lit(u8),
    I16Lit(i16),
    U16Lit(u16),
    I32Lit(i32),
    IntLit(u64),
    U32Lit(u32),
    I64Lit(i64),
    U64Lit(u64),
    StringLit(String),
    // Identifier
    Ident(String),
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Bang,
    AmpAmp,
    PipePipe,
    Shl,
    Shr,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    // Punctuation
    Dot,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semi,
    Arrow,
    Eq,
    // End
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Use => write!(f, "use"),
            Token::Kernel => write!(f, "kernel"),
            Token::Fn => write!(f, "fn"),
            Token::Let => write!(f, "let"),
            Token::Emit => write!(f, "emit"),
            Token::Return => write!(f, "return"),
            Token::While => write!(f, "while"),
            Token::BreakIf => write!(f, "break_if"),
            Token::Yield => write!(f, "yield"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Buffers => write!(f, "buffers"),
            Token::Textures => write!(f, "textures"),
            Token::Read => write!(f, "read"),
            Token::Write => write!(f, "write"),
            Token::Struct => write!(f, "struct"),
            Token::TyF32 => write!(f, "f32"),
            Token::TyF64 => write!(f, "f64"),
            Token::TyI8 => write!(f, "i8"),
            Token::TyU8 => write!(f, "u8"),
            Token::TyI16 => write!(f, "i16"),
            Token::TyU16 => write!(f, "u16"),
            Token::TyI32 => write!(f, "i32"),
            Token::TyU32 => write!(f, "u32"),
            Token::TyI64 => write!(f, "i64"),
            Token::TyU64 => write!(f, "u64"),
            Token::TyBool => write!(f, "bool"),
            Token::TyVec2 => write!(f, "vec2"),
            Token::TyVec3 => write!(f, "vec3"),
            Token::TyVec4 => write!(f, "vec4"),
            Token::TyMat2 => write!(f, "mat2"),
            Token::TyMat3 => write!(f, "mat3"),
            Token::TyMat4 => write!(f, "mat4"),
            Token::F32Lit(v) => write!(f, "{}f32", v),
            Token::FloatLit(v) => write!(f, "{}", v),
            Token::I8Lit(v) => write!(f, "{}i8", v),
            Token::U8Lit(v) => write!(f, "{}u8", v),
            Token::I16Lit(v) => write!(f, "{}i16", v),
            Token::U16Lit(v) => write!(f, "{}u16", v),
            Token::I32Lit(v) => write!(f, "{}i32", v),
            Token::IntLit(v) => write!(f, "{}", v),
            Token::U32Lit(v) => write!(f, "{}u32", v),
            Token::I64Lit(v) => write!(f, "{}i64", v),
            Token::U64Lit(v) => write!(f, "{}u64", v),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Amp => write!(f, "&"),
            Token::Pipe => write!(f, "|"),
            Token::Caret => write!(f, "^"),
            Token::Tilde => write!(f, "~"),
            Token::Bang => write!(f, "!"),
            Token::AmpAmp => write!(f, "&&"),
            Token::PipePipe => write!(f, "||"),
            Token::Shl => write!(f, "<<"),
            Token::Shr => write!(f, ">>"),
            Token::EqEq => write!(f, "=="),
            Token::BangEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Le => write!(f, "<="),
            Token::Gt => write!(f, ">"),
            Token::Ge => write!(f, ">="),
            Token::Dot => write!(f, "."),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semi => write!(f, ";"),
            Token::Arrow => write!(f, "->"),
            Token::Eq => write!(f, "="),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub line: usize,
    pub col: usize,
}

pub fn lex(input: &str) -> Result<Vec<Spanned>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;

    while i < chars.len() {
        let ch = chars[i];

        // Whitespace
        if ch == '\n' {
            line += 1;
            col = 1;
            i += 1;
            continue;
        }
        if ch.is_ascii_whitespace() {
            col += 1;
            i += 1;
            continue;
        }

        // Line comment
        if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        let start_col = col;

        // Number literal
        if ch.is_ascii_digit() || (ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            let mut has_dot = false;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                if chars[i] == '.' {
                    if has_dot {
                        break;
                    }
                    has_dot = true;
                }
                i += 1;
                col += 1;
            }
            // Check for type suffixes: 3-char (u32, i32, f32, f64, i16, u16, i64, u64) then 2-char (i8, u8)
            let suffix3 = if i + 2 < chars.len() { Some([chars[i], chars[i + 1], chars[i + 2]]) } else { None };
            let suffix2 = if i + 1 < chars.len() { Some([chars[i], chars[i + 1]]) } else { None };
            let text: String = chars[start..i].iter().collect();
            if !has_dot && suffix3 == Some(['u', '3', '2']) {
                i += 3;
                col += 3;
                let val = text.parse::<u32>().map_err(|e| {
                    format!("{}:{}: invalid u32 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::U32Lit(val), line, col: start_col });
            } else if !has_dot && suffix3 == Some(['i', '3', '2']) {
                i += 3;
                col += 3;
                let val = text.parse::<i32>().map_err(|e| {
                    format!("{}:{}: invalid i32 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::I32Lit(val), line, col: start_col });
            } else if !has_dot && suffix3 == Some(['i', '1', '6']) {
                i += 3;
                col += 3;
                let val = text.parse::<i16>().map_err(|e| {
                    format!("{}:{}: invalid i16 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::I16Lit(val), line, col: start_col });
            } else if !has_dot && suffix3 == Some(['u', '1', '6']) {
                i += 3;
                col += 3;
                let val = text.parse::<u16>().map_err(|e| {
                    format!("{}:{}: invalid u16 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::U16Lit(val), line, col: start_col });
            } else if !has_dot && suffix3 == Some(['i', '6', '4']) {
                i += 3;
                col += 3;
                let val = text.parse::<i64>().map_err(|e| {
                    format!("{}:{}: invalid i64 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::I64Lit(val), line, col: start_col });
            } else if !has_dot && suffix3 == Some(['u', '6', '4']) {
                i += 3;
                col += 3;
                let val = text.parse::<u64>().map_err(|e| {
                    format!("{}:{}: invalid u64 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::U64Lit(val), line, col: start_col });
            } else if suffix3 == Some(['f', '3', '2']) {
                i += 3;
                col += 3;
                let val = text.parse::<f32>().map_err(|e| {
                    format!("{}:{}: invalid f32 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::F32Lit(val), line, col: start_col });
            } else if suffix3 == Some(['f', '6', '4']) {
                i += 3;
                col += 3;
                let val = text.parse::<f64>().map_err(|e| {
                    format!("{}:{}: invalid f64 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::FloatLit(val), line, col: start_col });
            } else if !has_dot && suffix2 == Some(['i', '8']) {
                i += 2;
                col += 2;
                let val = text.parse::<i8>().map_err(|e| {
                    format!("{}:{}: invalid i8 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::I8Lit(val), line, col: start_col });
            } else if !has_dot && suffix2 == Some(['u', '8']) {
                i += 2;
                col += 2;
                let val = text.parse::<u8>().map_err(|e| {
                    format!("{}:{}: invalid u8 literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::U8Lit(val), line, col: start_col });
            } else if has_dot {
                let val = text.parse::<f64>().map_err(|e| {
                    format!("{}:{}: invalid float literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::FloatLit(val), line, col: start_col });
            } else {
                let val = text.parse::<u64>().map_err(|e| {
                    format!("{}:{}: invalid integer literal '{}': {}", line, start_col, text, e)
                })?;
                tokens.push(Spanned { token: Token::IntLit(val), line, col: start_col });
            }
            continue;
        }

        // Identifier or keyword
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
                col += 1;
            }
            let text: String = chars[start..i].iter().collect();
            let token = match text.as_str() {
                "use" => Token::Use,
                "kernel" => Token::Kernel,
                "fn" => Token::Fn,
                "let" => Token::Let,
                "emit" => Token::Emit,
                "return" => Token::Return,
                "while" => Token::While,
                "break_if" => Token::BreakIf,
                "yield" => Token::Yield,
                "if" => Token::If,
                "else" => Token::Else,
                // "as" is no longer a keyword — casts use type(expr) syntax
                "true" => Token::True,
                "false" => Token::False,
                "buffers" => Token::Buffers,
                "textures" => Token::Textures,
                "read" => Token::Read,
                "write" => Token::Write,
                "struct" => Token::Struct,
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
                _ => Token::Ident(text),
            };
            tokens.push(Spanned { token, line, col: start_col });
            continue;
        }

        // String literal
        if ch == '"' {
            i += 1;
            col += 1;
            let mut s = String::new();
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\n' {
                    return Err(format!("{}:{}: unterminated string literal", line, start_col));
                }
                s.push(chars[i]);
                i += 1;
                col += 1;
            }
            if i >= chars.len() {
                return Err(format!("{}:{}: unterminated string literal", line, start_col));
            }
            i += 1; // closing "
            col += 1;
            tokens.push(Spanned { token: Token::StringLit(s), line, col: start_col });
            continue;
        }

        // Multi-char operators
        let next = if i + 1 < chars.len() { Some(chars[i + 1]) } else { None };
        let (token, len) = match (ch, next) {
            ('-', Some('>')) => (Token::Arrow, 2),
            ('&', Some('&')) => (Token::AmpAmp, 2),
            ('|', Some('|')) => (Token::PipePipe, 2),
            ('<', Some('<')) => (Token::Shl, 2),
            ('<', Some('=')) => (Token::Le, 2),
            ('>', Some('>')) => (Token::Shr, 2),
            ('>', Some('=')) => (Token::Ge, 2),
            ('=', Some('=')) => (Token::EqEq, 2),
            ('!', Some('=')) => (Token::BangEq, 2),
            ('+', _) => (Token::Plus, 1),
            ('-', _) => (Token::Minus, 1),
            ('*', _) => (Token::Star, 1),
            ('/', _) => (Token::Slash, 1),
            ('%', _) => (Token::Percent, 1),
            ('&', _) => (Token::Amp, 1),
            ('|', _) => (Token::Pipe, 1),
            ('^', _) => (Token::Caret, 1),
            ('~', _) => (Token::Tilde, 1),
            ('!', _) => (Token::Bang, 1),
            ('<', _) => (Token::Lt, 1),
            ('>', _) => (Token::Gt, 1),
            ('=', _) => (Token::Eq, 1),
            ('(', _) => (Token::LParen, 1),
            (')', _) => (Token::RParen, 1),
            ('{', _) => (Token::LBrace, 1),
            ('}', _) => (Token::RBrace, 1),
            (',', _) => (Token::Comma, 1),
            (':', _) => (Token::Colon, 1),
            (';', _) => (Token::Semi, 1),
            ('.', _) => (Token::Dot, 1),
            _ => return Err(format!("{}:{}: unexpected character '{}'", line, col, ch)),
        };
        tokens.push(Spanned { token, line, col: start_col });
        i += len;
        col += len;
    }

    tokens.push(Spanned { token: Token::Eof, line, col });
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_basic() {
        let tokens = lex("let x = 1.0 + 2u32;").unwrap();
        assert!(matches!(tokens[0].token, Token::Let));
        assert!(matches!(tokens[1].token, Token::Ident(_)));
        assert!(matches!(tokens[2].token, Token::Eq));
        assert!(matches!(tokens[3].token, Token::FloatLit(_)));
        assert!(matches!(tokens[4].token, Token::Plus));
        assert!(matches!(tokens[5].token, Token::U32Lit(2)));
        assert!(matches!(tokens[6].token, Token::Semi));
    }

    #[test]
    fn lex_comment() {
        let tokens = lex("let x = 1; // comment\nlet y = 2;").unwrap();
        // Should have: let x = 1 ; let y = 2 ; EOF
        assert_eq!(tokens.iter().filter(|t| matches!(t.token, Token::Let)).count(), 2);
    }
}
