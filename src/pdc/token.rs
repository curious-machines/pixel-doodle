use super::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    StringLit(String),

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
    Import,
    From,
    Struct,
    Enum,
    Type,
    Pub,

    // Operators
    Plus,
    Minus,
    Star,
    StarStar, // **
    Slash,
    Percent,
    Eq,       // =
    EqEq,     // ==
    BangEq,   // !=
    Lt,       // <
    LtEq,     // <=
    LtLt,     // <<
    Gt,       // >
    GtEq,     // >=
    GtGt,     // >>
    Amp,      // &
    AmpAmp,   // &&
    Pipe,     // |
    PipePipe, // ||
    Caret,    // ^
    Tilde,    // ~
    Bang,     // !

    // Assignment operators
    PlusEq,
    MinusEq,
    StarEq,
    StarStarEq, // **=
    SlashEq,
    PercentEq,
    AmpEq,      // &=
    PipeEq,     // |=
    CaretEq,    // ^=
    LtLtEq,     // <<=
    GtGtEq,     // >>=

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    // Punctuation
    Question,  // ?
    Comma,
    Colon,
    Dot,
    Arrow,    // ->
    FatArrow, // =>
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
        "import" => Some(TokenKind::Import),
        "struct" => Some(TokenKind::Struct),
        "enum" => Some(TokenKind::Enum),
        "from" => Some(TokenKind::From),
        "type" => Some(TokenKind::Type),
        "pub" => Some(TokenKind::Pub),
        "true" => Some(TokenKind::BoolLit(true)),
        "false" => Some(TokenKind::BoolLit(false)),
        _ => None,
    }
}
