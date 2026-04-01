use super::error::PdcError;
use super::span::Span;
use super::token::{keyword_lookup, Token, TokenKind};

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
                if self.peek() == Some(b'=') {
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
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            b'>' => {
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            b'&' if self.peek() == Some(b'&') => {
                self.advance();
                TokenKind::AmpAmp
            }
            b'|' if self.peek() == Some(b'|') => {
                self.advance();
                TokenKind::PipePipe
            }
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

            let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap();
            let val: f64 = text
                .parse()
                .map_err(|_| self.err(start, format!("invalid float literal '{text}'")))?;
            return Ok(Token {
                kind: TokenKind::FloatLit(val),
                span: self.span_from(start),
            });
        }

        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap();
        let val: i64 = text
            .parse()
            .map_err(|_| self.err(start, format!("invalid integer literal '{text}'")))?;
        Ok(Token {
            kind: TokenKind::IntLit(val),
            span: self.span_from(start),
        })
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

        // Store as i64 (the type checker will determine the actual type)
        Ok(Token {
            kind: TokenKind::IntLit(val as i64),
            span: self.span_from(start),
        })
    }
}
