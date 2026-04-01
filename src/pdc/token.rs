use super::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),

    // Identifier
    Ident(String),

    // Keywords
    Const,
    Var,
    Builtin,
    Fn,
    Return,
    If,
    Else,
    Elsif,
    For,
    In,
    While,
    Loop,
    Break,
    Continue,
    Match,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,       // =
    EqEq,     // ==
    BangEq,   // !=
    Lt,       // <
    LtEq,     // <=
    Gt,       // >
    GtEq,     // >=
    AmpAmp,   // &&
    PipePipe, // ||
    Bang,     // !

    // Assignment operators
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    // Punctuation
    Comma,
    Colon,
    Dot,
    Arrow,    // ->
    DotDot,   // ..
    DotDotEq, // ..=

    // End of file
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub fn keyword_lookup(s: &str) -> Option<TokenKind> {
    match s {
        "const" => Some(TokenKind::Const),
        "var" => Some(TokenKind::Var),
        "builtin" => Some(TokenKind::Builtin),
        "fn" => Some(TokenKind::Fn),
        "return" => Some(TokenKind::Return),
        "if" => Some(TokenKind::If),
        "else" => Some(TokenKind::Else),
        "elsif" => Some(TokenKind::Elsif),
        "for" => Some(TokenKind::For),
        "in" => Some(TokenKind::In),
        "while" => Some(TokenKind::While),
        "loop" => Some(TokenKind::Loop),
        "break" => Some(TokenKind::Break),
        "continue" => Some(TokenKind::Continue),
        "match" => Some(TokenKind::Match),
        "true" => Some(TokenKind::BoolLit(true)),
        "false" => Some(TokenKind::BoolLit(false)),
        _ => None,
    }
}
