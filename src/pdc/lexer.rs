use super::error::PdcError;
use super::span::Span;
use super::token::{keyword_lookup, NumericSuffix, Token, TokenKind};

pub fn lex(source: &str) -> Result<Vec<Token>, PdcError> {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    loop {
        let tok = lexer.next_token()?;
        let is_eof = tok.kind == TokenKind::Eof;
        tokens.push(tok);
        if is_eof {
            break;
        }
    }
    Ok(tokens)
}

struct Lexer<'a> {
    source: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.source.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> u8 {
        let ch = self.source[self.pos];
        self.pos += 1;
        ch
    }

    fn span_from(&self, start: usize) -> Span {
        Span::new(start as u32, self.pos as u32)
    }

    fn err(&self, start: usize, msg: impl Into<String>) -> PdcError {
        PdcError::Lex {
            span: self.span_from(start),
            message: msg.into(),
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while let Some(ch) = self.peek() {
                if ch == b' ' || ch == b'\t' || ch == b'\n' || ch == b'\r' {
                    self.pos += 1;
                } else {
                    break;
                }
            }

            // Skip line comments
            if self.peek() == Some(b'/') && self.peek2() == Some(b'/') {
                while let Some(ch) = self.peek() {
                    self.pos += 1;
                    if ch == b'\n' {
                        break;
                    }
                }
                continue;
            }

            // Skip block comments (nestable)
            if self.peek() == Some(b'/') && self.peek2() == Some(b'*') {
                self.pos += 2; // consume /*
                let mut depth = 1u32;
                while depth > 0 {
                    match self.peek() {
                        None => break, // unterminated — let the parser deal with it
                        Some(b'/') if self.peek2() == Some(b'*') => {
                            self.pos += 2;
                            depth += 1;
                        }
                        Some(b'*') if self.peek2() == Some(b'/') => {
                            self.pos += 2;
                            depth -= 1;
                        }
                        _ => {
                            self.pos += 1;
                        }
                    }
                }
                continue;
            }

            break;
        }
    }

    fn next_token(&mut self) -> Result<Token, PdcError> {
        self.skip_whitespace_and_comments();

        let start = self.pos;

        let Some(ch) = self.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                span: self.span_from(start),
            });
        };

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == b'_' {
            return self.lex_ident(start);
        }

        // String literals
        if ch == b'"' {
            return self.lex_string(start);
        }

        // Numbers (including hex)
        if ch.is_ascii_digit() {
            return self.lex_number(start);
        }

        // Operators and punctuation
        self.advance();
        let kind = match ch {
            b'+' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::PlusEq
                } else {
                    TokenKind::Plus
                }
            }
            b'-' => {
                if self.peek() == Some(b'>') {
                    self.advance();
                    TokenKind::Arrow
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::MinusEq
                } else {
                    TokenKind::Minus
                }
            }
            b'*' => {
                if self.peek() == Some(b'*') {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        TokenKind::StarStarEq
                    } else {
                        TokenKind::StarStar
                    }
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::StarEq
                } else {
                    TokenKind::Star
                }
            }
            b'/' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::SlashEq
                } else {
                    TokenKind::Slash
                }
            }
            b'%' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::PercentEq
                } else {
                    TokenKind::Percent
                }
            }
            b'=' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::EqEq
                } else if self.peek() == Some(b'>') {
                    self.advance();
                    TokenKind::FatArrow
                } else {
                    TokenKind::Eq
                }
            }
            b'!' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::BangEq
                } else {
                    TokenKind::Bang
                }
            }
            b'<' => {
                if self.peek() == Some(b'<') {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        TokenKind::LtLtEq
                    } else {
                        TokenKind::LtLt
                    }
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            b'>' => {
                if self.peek() == Some(b'>') {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        TokenKind::GtGtEq
                    } else {
                        TokenKind::GtGt
                    }
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            b'&' => {
                if self.peek() == Some(b'&') {
                    self.advance();
                    TokenKind::AmpAmp
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::AmpEq
                } else {
                    TokenKind::Amp
                }
            }
            b'|' => {
                if self.peek() == Some(b'|') {
                    self.advance();
                    TokenKind::PipePipe
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::PipeEq
                } else {
                    TokenKind::Pipe
                }
            }
            b'^' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::CaretEq
                } else {
                    TokenKind::Caret
                }
            }
            b'~' => TokenKind::Tilde,
            b'?' => TokenKind::Question,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b',' => TokenKind::Comma,
            b':' => TokenKind::Colon,
            b'.' => {
                if self.peek() == Some(b'.') {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        TokenKind::DotDotEq
                    } else {
                        TokenKind::DotDot
                    }
                } else {
                    TokenKind::Dot
                }
            }
            _ => return Err(self.err(start, format!("unexpected character '{}'", ch as char))),
        };

        Ok(Token {
            kind,
            span: self.span_from(start),
        })
    }

    fn lex_ident(&mut self, start: usize) -> Result<Token, PdcError> {
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap();
        let kind = keyword_lookup(text).unwrap_or_else(|| TokenKind::Ident(text.to_string()));

        Ok(Token {
            kind,
            span: self.span_from(start),
        })
    }

    fn lex_number(&mut self, start: usize) -> Result<Token, PdcError> {
        // Check for hex: 0x or 0X
        if self.peek() == Some(b'0') && matches!(self.peek2(), Some(b'x' | b'X')) {
            self.advance(); // consume '0'
            self.advance(); // consume 'x'
            return self.lex_hex(start);
        }

        // Decimal integer or float
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        // Check for decimal point (but not `..` range operator)
        if self.peek() == Some(b'.') && self.peek2() != Some(b'.') {
            self.advance(); // consume '.'
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }

            // Scientific notation
            if matches!(self.peek(), Some(b'e' | b'E')) {
                self.advance();
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.advance();
                }
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }

            let num_end = self.pos;
            let suffix = self.lex_numeric_suffix(start)?;

            // Float literals cannot have integer suffixes
            if let Some(ref s) = suffix {
                if !s.is_float() {
                    return Err(self.err(start, format!(
                        "float literal cannot have integer suffix '{:?}'",
                        s
                    )));
                }
            }

            let text = std::str::from_utf8(&self.source[start..num_end]).unwrap();
            let val: f64 = text
                .parse()
                .map_err(|_| self.err(start, format!("invalid float literal '{text}'")))?;
            return Ok(Token {
                kind: TokenKind::FloatLit(val, suffix),
                span: self.span_from(start),
            });
        }

        let num_end = self.pos;
        let suffix = self.lex_numeric_suffix(start)?;

        let text = std::str::from_utf8(&self.source[start..num_end]).unwrap();

        // Integer literal with float suffix becomes a FloatLit
        if let Some(ref s) = suffix {
            if s.is_float() {
                let val: f64 = text
                    .parse()
                    .map_err(|_| self.err(start, format!("invalid numeric literal '{text}'")))?;
                return Ok(Token {
                    kind: TokenKind::FloatLit(val, suffix),
                    span: self.span_from(start),
                });
            }
        }

        let val: i64 = text
            .parse()
            .map_err(|_| self.err(start, format!("invalid integer literal '{text}'")))?;
        Ok(Token {
            kind: TokenKind::IntLit(val, suffix),
            span: self.span_from(start),
        })
    }

    fn lex_string(&mut self, start: usize) -> Result<Token, PdcError> {
        self.advance(); // consume opening '"'
        let mut value = String::new();
        loop {
            match self.peek() {
                None => return Err(self.err(start, "unterminated string literal")),
                Some(b'"') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance();
                    match self.peek() {
                        Some(b'n') => { self.advance(); value.push('\n'); }
                        Some(b't') => { self.advance(); value.push('\t'); }
                        Some(b'r') => { self.advance(); value.push('\r'); }
                        Some(b'\\') => { self.advance(); value.push('\\'); }
                        Some(b'"') => { self.advance(); value.push('"'); }
                        Some(b'0') => { self.advance(); value.push('\0'); }
                        Some(ch) => return Err(self.err(start, format!("unknown escape sequence '\\{}'", ch as char))),
                        None => return Err(self.err(start, "unterminated string literal")),
                    }
                }
                Some(ch) => {
                    self.advance();
                    value.push(ch as char);
                }
            }
        }
        Ok(Token {
            kind: TokenKind::StringLit(value),
            span: self.span_from(start),
        })
    }

    /// Try to consume a numeric type suffix (u8, u32, f32, etc.) after a number literal.
    fn lex_numeric_suffix(&mut self, start: usize) -> Result<Option<NumericSuffix>, PdcError> {
        // Check if the next character could start a suffix (f, i, u)
        let first = match self.peek() {
            Some(b'f' | b'i' | b'u') => self.source[self.pos] as char,
            _ => return Ok(None),
        };

        // Peek ahead to collect potential suffix characters (letters + digits)
        let suffix_start = self.pos;
        let mut suffix_end = self.pos + 1;
        while suffix_end < self.source.len() && (self.source[suffix_end].is_ascii_alphanumeric() || self.source[suffix_end] == b'_') {
            suffix_end += 1;
        }

        let candidate = std::str::from_utf8(&self.source[suffix_start..suffix_end]).unwrap();

        // Only consume if it's a valid numeric suffix — otherwise leave it as an identifier
        if let Some(suffix) = NumericSuffix::from_str(candidate) {
            // Make sure the suffix isn't part of a longer identifier (e.g., `42u8foo`)
            if suffix_end < self.source.len() && (self.source[suffix_end].is_ascii_alphanumeric() || self.source[suffix_end] == b'_') {
                return Err(self.err(start, format!(
                    "invalid suffix '{}' on numeric literal",
                    std::str::from_utf8(&self.source[suffix_start..suffix_end + 1]).unwrap_or("?")
                )));
            }
            // Consume the suffix
            while self.pos < suffix_end {
                self.advance();
            }
            Ok(Some(suffix))
        } else {
            // Not a valid suffix — could be something like `42if` which should be
            // the number 42 followed by the keyword `if`. Don't consume.
            // But first check: if it starts with a suffix letter followed by digits,
            // it might be a typo like `u33` — error on that.
            if matches!(first, 'f' | 'i' | 'u') && candidate.len() >= 2 && candidate[1..].chars().all(|c| c.is_ascii_digit()) {
                return Err(self.err(start, format!(
                    "invalid numeric suffix '{candidate}' (valid suffixes: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64)"
                )));
            }
            Ok(None)
        }
    }

    fn lex_hex(&mut self, start: usize) -> Result<Token, PdcError> {
        let hex_start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_hexdigit() {
                self.advance();
            } else {
                break;
            }
        }

        if self.pos == hex_start {
            return Err(self.err(start, "expected hex digits after '0x'"));
        }

        let hex_text = std::str::from_utf8(&self.source[hex_start..self.pos]).unwrap();
        // Parse as u64 first to handle values like 0xFFFF8800
        let val = u64::from_str_radix(hex_text, 16)
            .map_err(|_| self.err(start, format!("invalid hex literal '0x{hex_text}'")))?;

        let suffix = self.lex_numeric_suffix(start)?;

        // Hex literals with float suffix don't make sense
        if let Some(ref s) = suffix {
            if s.is_float() {
                return Err(self.err(start, format!(
                    "hex literal cannot have float suffix '{:?}'", s
                )));
            }
        }

        // Store as i64 (the type checker will determine the actual type)
        Ok(Token {
            kind: TokenKind::IntLit(val as i64, suffix),
            span: self.span_from(start),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: lex source and return just the TokenKinds (excluding Eof).
    fn kinds(source: &str) -> Vec<TokenKind> {
        let tokens = lex(source).expect("lex failed");
        tokens.into_iter().map(|t| t.kind).filter(|k| *k != TokenKind::Eof).collect()
    }

    /// Helper: lex source and expect a single non-Eof token, returning its kind.
    fn single(source: &str) -> TokenKind {
        let k = kinds(source);
        assert_eq!(k.len(), 1, "expected 1 token, got {}: {:?}", k.len(), k);
        k.into_iter().next().unwrap()
    }

    /// Helper: lex source and expect an error containing the given substring.
    fn lex_err(source: &str, expected_substr: &str) {
        match lex(source) {
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains(expected_substr),
                    "error '{}' does not contain '{}'", msg, expected_substr);
            }
            Ok(tokens) => panic!("expected error containing '{}', got {:?}", expected_substr, tokens),
        }
    }

    // ---- Integer literals ----

    #[test]
    fn integer_zero() {
        assert_eq!(single("0"), TokenKind::IntLit(0, None));
    }

    #[test]
    fn integer_positive() {
        assert_eq!(single("42"), TokenKind::IntLit(42, None));
    }

    #[test]
    fn integer_large() {
        assert_eq!(single("2147483647"), TokenKind::IntLit(2147483647, None));
    }

    #[test]
    fn hex_literal() {
        assert_eq!(single("0xFF"), TokenKind::IntLit(0xFF, None));
    }

    #[test]
    fn hex_uppercase() {
        assert_eq!(single("0XAB"), TokenKind::IntLit(0xAB, None));
    }

    #[test]
    fn hex_large() {
        assert_eq!(single("0xDEADBEEF"), TokenKind::IntLit(0xDEADBEEFu64 as i64, None));
    }

    #[test]
    fn hex_no_digits() {
        lex_err("0x", "expected hex digits");
    }

    #[test]
    fn hex_no_digits_followed_by_space() {
        lex_err("0x ", "expected hex digits");
    }

    // ---- Float literals ----

    #[test]
    fn float_simple() {
        assert_eq!(single("3.14"), TokenKind::FloatLit(3.14, None));
    }

    #[test]
    fn float_leading_zero() {
        assert_eq!(single("0.5"), TokenKind::FloatLit(0.5, None));
    }

    #[test]
    fn float_scientific() {
        assert_eq!(single("1.0e5"), TokenKind::FloatLit(1.0e5, None));
    }

    #[test]
    fn float_scientific_negative_exp() {
        assert_eq!(single("2.5e-3"), TokenKind::FloatLit(2.5e-3, None));
    }

    #[test]
    fn float_scientific_positive_exp() {
        assert_eq!(single("1.0E+2"), TokenKind::FloatLit(1.0e+2, None));
    }

    #[test]
    fn float_no_fractional_digits() {
        // "1." followed by non-digit — should parse as 1.0
        assert_eq!(single("1.0"), TokenKind::FloatLit(1.0, None));
    }

    // ---- Integer vs range disambiguation ----

    #[test]
    fn integer_before_range() {
        // "0..10" should lex as IntLit(0), DotDot, IntLit(10)
        let k = kinds("0..10");
        assert_eq!(k, vec![TokenKind::IntLit(0, None), TokenKind::DotDot, TokenKind::IntLit(10, None)]);
    }

    // ---- String literals ----

    #[test]
    fn string_simple() {
        assert_eq!(single("\"hello\""), TokenKind::StringLit("hello".to_string()));
    }

    #[test]
    fn string_empty() {
        assert_eq!(single("\"\""), TokenKind::StringLit(String::new()));
    }

    #[test]
    fn string_escapes() {
        assert_eq!(single("\"\\n\\t\\r\\\\\\\"\\0\""),
            TokenKind::StringLit("\n\t\r\\\"\0".to_string()));
    }

    #[test]
    fn string_unterminated() {
        lex_err("\"hello", "unterminated string");
    }

    #[test]
    fn string_unknown_escape() {
        lex_err("\"\\q\"", "unknown escape");
    }

    #[test]
    fn string_unterminated_after_escape() {
        lex_err("\"\\", "unterminated string");
    }

    // ---- Boolean literals ----

    #[test]
    fn bool_true() {
        assert_eq!(single("true"), TokenKind::BoolLit(true));
    }

    #[test]
    fn bool_false() {
        assert_eq!(single("false"), TokenKind::BoolLit(false));
    }

    // ---- Identifiers ----

    #[test]
    fn ident_simple() {
        assert_eq!(single("foo"), TokenKind::Ident("foo".to_string()));
    }

    #[test]
    fn ident_with_underscore() {
        assert_eq!(single("_bar_42"), TokenKind::Ident("_bar_42".to_string()));
    }

    #[test]
    fn ident_starting_with_underscore() {
        assert_eq!(single("_"), TokenKind::Ident("_".to_string()));
    }

    // ---- Keywords ----

    #[test]
    fn keywords() {
        let cases = vec![
            ("const", TokenKind::Const),
            ("var", TokenKind::Var),
            ("builtin", TokenKind::Builtin),
            ("fn", TokenKind::Fn),
            ("return", TokenKind::Return),
            ("if", TokenKind::If),
            ("else", TokenKind::Else),
            ("elsif", TokenKind::Elsif),
            ("for", TokenKind::For),
            ("in", TokenKind::In),
            ("while", TokenKind::While),
            ("loop", TokenKind::Loop),
            ("break", TokenKind::Break),
            ("continue", TokenKind::Continue),
            ("match", TokenKind::Match),
            ("import", TokenKind::Import),
            ("from", TokenKind::From),
            ("struct", TokenKind::Struct),
            ("enum", TokenKind::Enum),
            ("type", TokenKind::Type),
            ("pub", TokenKind::Pub),
            ("operator", TokenKind::Operator),
        ];
        for (src, expected) in cases {
            assert_eq!(single(src), expected, "keyword: {}", src);
        }
    }

    // ---- Operators ----

    #[test]
    fn single_char_operators() {
        let cases = vec![
            ("+", TokenKind::Plus),
            ("-", TokenKind::Minus),
            ("*", TokenKind::Star),
            ("/", TokenKind::Slash),
            ("%", TokenKind::Percent),
            ("=", TokenKind::Eq),
            ("!", TokenKind::Bang),
            ("<", TokenKind::Lt),
            (">", TokenKind::Gt),
            ("&", TokenKind::Amp),
            ("|", TokenKind::Pipe),
            ("^", TokenKind::Caret),
            ("~", TokenKind::Tilde),
            ("?", TokenKind::Question),
            (".", TokenKind::Dot),
        ];
        for (src, expected) in cases {
            assert_eq!(single(src), expected, "operator: {}", src);
        }
    }

    #[test]
    fn multi_char_operators() {
        let cases = vec![
            ("+=", TokenKind::PlusEq),
            ("-=", TokenKind::MinusEq),
            ("*=", TokenKind::StarEq),
            ("**", TokenKind::StarStar),
            ("**=", TokenKind::StarStarEq),
            ("/=", TokenKind::SlashEq),
            ("%=", TokenKind::PercentEq),
            ("==", TokenKind::EqEq),
            ("!=", TokenKind::BangEq),
            ("<=", TokenKind::LtEq),
            (">=", TokenKind::GtEq),
            ("<<", TokenKind::LtLt),
            (">>", TokenKind::GtGt),
            ("<<=", TokenKind::LtLtEq),
            (">>=", TokenKind::GtGtEq),
            ("&&", TokenKind::AmpAmp),
            ("||", TokenKind::PipePipe),
            ("&=", TokenKind::AmpEq),
            ("|=", TokenKind::PipeEq),
            ("^=", TokenKind::CaretEq),
            ("->", TokenKind::Arrow),
            ("=>", TokenKind::FatArrow),
            ("..", TokenKind::DotDot),
            ("..=", TokenKind::DotDotEq),
        ];
        for (src, expected) in cases {
            assert_eq!(single(src), expected, "operator: {}", src);
        }
    }

    // ---- Delimiters ----

    #[test]
    fn delimiters() {
        let cases = vec![
            ("(", TokenKind::LParen),
            (")", TokenKind::RParen),
            ("{", TokenKind::LBrace),
            ("}", TokenKind::RBrace),
            ("[", TokenKind::LBracket),
            ("]", TokenKind::RBracket),
            (",", TokenKind::Comma),
            (":", TokenKind::Colon),
        ];
        for (src, expected) in cases {
            assert_eq!(single(src), expected, "delimiter: {}", src);
        }
    }

    // ---- Comments ----

    #[test]
    fn line_comment_skipped() {
        assert_eq!(kinds("42 // this is a comment"), vec![TokenKind::IntLit(42, None)]);
    }

    #[test]
    fn block_comment_skipped() {
        assert_eq!(kinds("42 /* block */ 7"), vec![TokenKind::IntLit(42, None), TokenKind::IntLit(7, None)]);
    }

    #[test]
    fn nested_block_comment() {
        assert_eq!(kinds("1 /* outer /* inner */ still comment */ 2"),
            vec![TokenKind::IntLit(1, None), TokenKind::IntLit(2, None)]);
    }

    #[test]
    fn comment_only_source() {
        assert_eq!(kinds("// just a comment"), vec![]);
    }

    #[test]
    fn block_comment_only() {
        assert_eq!(kinds("/* nothing */"), vec![]);
    }

    // ---- Whitespace ----

    #[test]
    fn empty_source() {
        assert_eq!(kinds(""), vec![]);
    }

    #[test]
    fn whitespace_only() {
        assert_eq!(kinds("   \t\n\r  "), vec![]);
    }

    // ---- Multi-token sequences ----

    #[test]
    fn adjacent_tokens_no_whitespace() {
        assert_eq!(kinds("123abc"), vec![TokenKind::IntLit(123, None), TokenKind::Ident("abc".to_string())]);
    }

    #[test]
    fn function_declaration() {
        let k = kinds("fn add(a: f64, b: f64) -> f64");
        assert_eq!(k, vec![
            TokenKind::Fn,
            TokenKind::Ident("add".to_string()),
            TokenKind::LParen,
            TokenKind::Ident("a".to_string()),
            TokenKind::Colon,
            TokenKind::Ident("f64".to_string()),
            TokenKind::Comma,
            TokenKind::Ident("b".to_string()),
            TokenKind::Colon,
            TokenKind::Ident("f64".to_string()),
            TokenKind::RParen,
            TokenKind::Arrow,
            TokenKind::Ident("f64".to_string()),
        ]);
    }

    #[test]
    fn expression_tokens() {
        let k = kinds("x + 3.0 * y");
        assert_eq!(k, vec![
            TokenKind::Ident("x".to_string()),
            TokenKind::Plus,
            TokenKind::FloatLit(3.0, None),
            TokenKind::Star,
            TokenKind::Ident("y".to_string()),
        ]);
    }

    // ---- Error cases ----

    #[test]
    fn unexpected_character() {
        lex_err("@", "unexpected character");
    }

    #[test]
    fn unexpected_character_backtick() {
        lex_err("`", "unexpected character");
    }

    // ---- Numeric suffixes ----

    #[test]
    fn int_suffix_u8() {
        assert_eq!(single("255u8"), TokenKind::IntLit(255, Some(NumericSuffix::U8)));
    }

    #[test]
    fn int_suffix_u16() {
        assert_eq!(single("1000u16"), TokenKind::IntLit(1000, Some(NumericSuffix::U16)));
    }

    #[test]
    fn int_suffix_u32() {
        assert_eq!(single("42u32"), TokenKind::IntLit(42, Some(NumericSuffix::U32)));
    }

    #[test]
    fn int_suffix_u64() {
        assert_eq!(single("100u64"), TokenKind::IntLit(100, Some(NumericSuffix::U64)));
    }

    #[test]
    fn int_suffix_i8() {
        assert_eq!(single("127i8"), TokenKind::IntLit(127, Some(NumericSuffix::I8)));
    }

    #[test]
    fn int_suffix_i16() {
        assert_eq!(single("500i16"), TokenKind::IntLit(500, Some(NumericSuffix::I16)));
    }

    #[test]
    fn int_suffix_i32() {
        assert_eq!(single("42i32"), TokenKind::IntLit(42, Some(NumericSuffix::I32)));
    }

    #[test]
    fn int_suffix_i64() {
        assert_eq!(single("42i64"), TokenKind::IntLit(42, Some(NumericSuffix::I64)));
    }

    #[test]
    fn int_with_float_suffix_f32() {
        // Integer literal with f32 suffix becomes FloatLit
        assert_eq!(single("42f32"), TokenKind::FloatLit(42.0, Some(NumericSuffix::F32)));
    }

    #[test]
    fn int_with_float_suffix_f64() {
        assert_eq!(single("42f64"), TokenKind::FloatLit(42.0, Some(NumericSuffix::F64)));
    }

    #[test]
    fn float_suffix_f32() {
        assert_eq!(single("3.14f32"), TokenKind::FloatLit(3.14, Some(NumericSuffix::F32)));
    }

    #[test]
    fn float_suffix_f64() {
        assert_eq!(single("3.14f64"), TokenKind::FloatLit(3.14, Some(NumericSuffix::F64)));
    }

    #[test]
    fn float_with_int_suffix_error() {
        lex_err("3.14u8", "float literal cannot have integer suffix");
    }

    #[test]
    fn float_with_i32_suffix_error() {
        lex_err("1.0i32", "float literal cannot have integer suffix");
    }

    #[test]
    fn hex_with_suffix_u8() {
        assert_eq!(single("0xFFu8"), TokenKind::IntLit(0xFF, Some(NumericSuffix::U8)));
    }

    #[test]
    fn hex_with_suffix_u32() {
        assert_eq!(single("0xDEADu32"), TokenKind::IntLit(0xDEADu64 as i64, Some(NumericSuffix::U32)));
    }

    #[test]
    fn hex_with_f_is_hex_digit() {
        // 'f' is a valid hex digit, so 0xFFf32 is parsed as hex 0xFFf32, not 0xFF with suffix f32
        assert_eq!(single("0xFFf32"), TokenKind::IntLit(0xFFf32, None));
    }

    #[test]
    fn invalid_suffix_error() {
        lex_err("42u33", "invalid numeric suffix");
    }

    #[test]
    fn suffix_not_confused_with_ident() {
        // `42if` should lex as IntLit(42) + keyword `if`, not a suffix error
        let k = kinds("42 if");
        assert_eq!(k, vec![TokenKind::IntLit(42, None), TokenKind::If]);
    }

    #[test]
    fn suffix_span_included() {
        let tokens = lex("42u8").unwrap();
        // "42u8" is 4 chars, span should be 0..4
        assert_eq!(tokens[0].span, Span::new(0, 4));
    }

    #[test]
    fn zero_with_suffix() {
        assert_eq!(single("0u8"), TokenKind::IntLit(0, Some(NumericSuffix::U8)));
    }

    #[test]
    fn suffix_followed_by_operator() {
        let k = kinds("42u8 + 1u8");
        assert_eq!(k, vec![
            TokenKind::IntLit(42, Some(NumericSuffix::U8)),
            TokenKind::Plus,
            TokenKind::IntLit(1, Some(NumericSuffix::U8)),
        ]);
    }

    #[test]
    fn int_before_range_with_suffix() {
        // "0u32..10u32" should lex correctly
        let k = kinds("0u32..10u32");
        assert_eq!(k, vec![
            TokenKind::IntLit(0, Some(NumericSuffix::U32)),
            TokenKind::DotDot,
            TokenKind::IntLit(10, Some(NumericSuffix::U32)),
        ]);
    }

    // ---- Span tracking ----

    #[test]
    fn span_positions() {
        let tokens = lex("ab 12").unwrap();
        // "ab" at 0..2, "12" at 3..5
        assert_eq!(tokens[0].span, Span::new(0, 2));
        assert_eq!(tokens[1].span, Span::new(3, 5));
    }
}
