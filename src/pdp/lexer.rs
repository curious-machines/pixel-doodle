use super::token::{Spanned, Token};

pub fn lex(input: &str) -> Result<Vec<Spanned>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;

    while i < chars.len() {
        let ch = chars[i];

        // Newline
        if ch == '\n' {
            line += 1;
            col = 1;
            i += 1;
            continue;
        }

        // Whitespace
        if ch.is_ascii_whitespace() {
            col += 1;
            i += 1;
            continue;
        }

        // Line comment: # to end of line
        if ch == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        let start_col = col;

        // Number literal
        if ch.is_ascii_digit()
            || (ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            let mut has_dot = false;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                if chars[i] == '.' {
                    // Check for '..' (range operator) — don't consume second dot
                    if i + 1 < chars.len() && chars[i + 1] == '.' {
                        break;
                    }
                    if has_dot {
                        break;
                    }
                    has_dot = true;
                }
                i += 1;
                col += 1;
            }
            let text: String = chars[start..i].iter().collect();
            if has_dot {
                let val = text.parse::<f64>().map_err(|e| {
                    format!("{line}:{start_col}: invalid float literal '{text}': {e}")
                })?;
                tokens.push(Spanned {
                    token: Token::FloatLit(val),
                    line,
                    col: start_col,
                });
            } else {
                let val = text.parse::<u64>().map_err(|e| {
                    format!("{line}:{start_col}: invalid integer literal '{text}': {e}")
                })?;
                tokens.push(Spanned {
                    token: Token::IntLit(val),
                    line,
                    col: start_col,
                });
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
                "pixel" => Token::Pixel,
                "init" => Token::Init,
                "kernel" => Token::Kernel,
                "buffer" => Token::Buffer,
                "constant" => Token::Constant,
                "pipeline" => Token::Pipeline,
                "run" => Token::Run,
                "display" => Token::Display,
                "swap" => Token::Swap,
                "loop" => Token::Loop,
                "accumulate" => Token::Accumulate,
                "on" => Token::On,
                "key" => Token::Key,
                "click" => Token::Click,
                "include" => Token::Include,
                "settings" => Token::Settings,
                "title" => Token::Title,
                "range" => Token::Range,
                "gpu" => Token::Gpu,
                "true" => Token::True,
                "false" => Token::False,
                _ => Token::Ident(text),
            };
            tokens.push(Spanned {
                token,
                line,
                col: start_col,
            });
            continue;
        }

        // String literal
        if ch == '"' {
            i += 1;
            col += 1;
            let mut s = String::new();
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\n' {
                    return Err(format!(
                        "{line}:{start_col}: unterminated string literal"
                    ));
                }
                s.push(chars[i]);
                i += 1;
                col += 1;
            }
            if i >= chars.len() {
                return Err(format!(
                    "{line}:{start_col}: unterminated string literal"
                ));
            }
            i += 1; // closing "
            col += 1;
            tokens.push(Spanned {
                token: Token::StringLit(s),
                line,
                col: start_col,
            });
            continue;
        }

        // Multi-char operators
        let next = if i + 1 < chars.len() {
            Some(chars[i + 1])
        } else {
            None
        };
        let next2 = if i + 2 < chars.len() {
            Some(chars[i + 2])
        } else {
            None
        };

        let (token, len) = match (ch, next, next2) {
            // <-> swap arrow (must check before < alone)
            ('<', Some('-'), Some('>')) => (Token::SwapArrow, 3),
            // Compound assignment
            ('+', Some('='), _) => (Token::PlusEq, 2),
            ('-', Some('='), _) => (Token::MinusEq, 2),
            ('*', Some('='), _) => (Token::StarEq, 2),
            ('/', Some('='), _) => (Token::SlashEq, 2),
            // Range
            ('.', Some('.'), _) => (Token::DotDot, 2),
            // Single-char
            ('=', _, _) => (Token::Eq, 1),
            ('+', _, _) => (Token::Plus, 1),
            ('-', _, _) => (Token::Minus, 1),
            ('*', _, _) => (Token::Star, 1),
            ('/', _, _) => (Token::Slash, 1),
            ('!', _, _) => (Token::Bang, 1),
            ('(', _, _) => (Token::LParen, 1),
            (')', _, _) => (Token::RParen, 1),
            ('{', _, _) => (Token::LBrace, 1),
            ('}', _, _) => (Token::RBrace, 1),
            (':', _, _) => (Token::Colon, 1),
            (',', _, _) => (Token::Comma, 1),
            ('<', _, _) => (Token::Lt, 1),
            ('>', _, _) => (Token::Gt, 1),
            _ => {
                return Err(format!(
                    "{line}:{col}: unexpected character '{ch}'"
                ))
            }
        };
        tokens.push(Spanned {
            token,
            line,
            col: start_col,
        });
        i += len;
        col += len;
    }

    tokens.push(Spanned {
        token: Token::Eof,
        line,
        col,
    });
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_kernel_decl() {
        let tokens = lex(r#"pixel kernel "gradient.pd""#).unwrap();
        assert!(matches!(tokens[0].token, Token::Pixel));
        assert!(matches!(tokens[1].token, Token::Kernel));
        assert!(matches!(tokens[2].token, Token::StringLit(_)));
        assert!(matches!(tokens[3].token, Token::Eof));
    }

    #[test]
    fn lex_buffer_constant() {
        let tokens = lex("buffer u = constant(1.0)").unwrap();
        assert!(matches!(tokens[0].token, Token::Buffer));
        assert!(matches!(tokens[1].token, Token::Ident(_)));
        assert!(matches!(tokens[2].token, Token::Eq));
        assert!(matches!(tokens[3].token, Token::Constant));
        assert!(matches!(tokens[4].token, Token::LParen));
        assert!(matches!(tokens[5].token, Token::FloatLit(_)));
        assert!(matches!(tokens[6].token, Token::RParen));
    }

    #[test]
    fn lex_swap_arrow() {
        let tokens = lex("swap u <-> v").unwrap();
        assert!(matches!(tokens[0].token, Token::Swap));
        assert!(matches!(tokens[1].token, Token::Ident(_)));
        assert!(matches!(tokens[2].token, Token::SwapArrow));
        assert!(matches!(tokens[3].token, Token::Ident(_)));
    }

    #[test]
    fn lex_compound_assignment() {
        let tokens = lex("center_x += 0.1").unwrap();
        assert!(matches!(tokens[0].token, Token::Ident(_)));
        assert!(matches!(tokens[1].token, Token::PlusEq));
        assert!(matches!(tokens[2].token, Token::FloatLit(_)));
    }

    #[test]
    fn lex_range() {
        let tokens = lex("iterations: range(1..10) = 8").unwrap();
        assert!(matches!(tokens[0].token, Token::Ident(_)));
        assert!(matches!(tokens[1].token, Token::Colon));
        assert!(matches!(tokens[2].token, Token::Range));
        assert!(matches!(tokens[3].token, Token::LParen));
        assert!(matches!(tokens[4].token, Token::IntLit(1)));
        assert!(matches!(tokens[5].token, Token::DotDot));
        assert!(matches!(tokens[6].token, Token::IntLit(10)));
        assert!(matches!(tokens[7].token, Token::RParen));
        assert!(matches!(tokens[8].token, Token::Eq));
        assert!(matches!(tokens[9].token, Token::IntLit(8)));
    }

    #[test]
    fn lex_comment() {
        let tokens = lex("# this is a comment\npixel kernel \"test.pd\"").unwrap();
        assert!(matches!(tokens[0].token, Token::Pixel));
    }

    #[test]
    fn lex_range_float() {
        let tokens = lex("range(1.0..20.0)").unwrap();
        assert!(matches!(tokens[0].token, Token::Range));
        assert!(matches!(tokens[1].token, Token::LParen));
        assert!(matches!(tokens[2].token, Token::FloatLit(v) if (v - 1.0).abs() < 1e-10));
        assert!(matches!(tokens[3].token, Token::DotDot));
        assert!(matches!(tokens[4].token, Token::FloatLit(v) if (v - 20.0).abs() < 1e-10));
        assert!(matches!(tokens[5].token, Token::RParen));
    }

    #[test]
    fn lex_pipeline_steps() {
        let tokens = lex("loop(iterations: 40) {\n  run jacobi\n}").unwrap();
        assert!(matches!(tokens[0].token, Token::Loop));
        assert!(matches!(tokens[1].token, Token::LParen));
    }

    #[test]
    fn lex_include() {
        let tokens = lex(r#"include "shared/pan_zoom.pdp""#).unwrap();
        assert!(matches!(tokens[0].token, Token::Include));
        assert!(matches!(tokens[1].token, Token::StringLit(_)));
    }

    #[test]
    fn lex_negative_number_in_assignment() {
        let tokens = lex("vy = run inject(value: -3.0)").unwrap();
        // -3.0 should be: Minus, FloatLit(3.0)
        // Find the minus token
        let minus_idx = tokens
            .iter()
            .position(|t| matches!(t.token, Token::Minus))
            .unwrap();
        assert!(matches!(tokens[minus_idx + 1].token, Token::FloatLit(_)));
    }
}
