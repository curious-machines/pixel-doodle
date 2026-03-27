use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Kernel type keywords
    Pixel,
    Init,
    Kernel,
    // Buffer
    Buffer,
    Constant,
    // Declaration keywords
    Builtin,
    Var,
    Const,
    // Pipeline
    Pipeline,
    Run,
    Display,
    Swap,
    Loop,
    Accumulate,
    // Events
    On,
    Keydown,
    Keypress,
    Keyup,
    Mousedown,
    Click,
    Mouseup,
    // Buffer bindings keyword
    With,
    // Include
    Include,
    // Settings
    Settings,
    Title,
    Range,
    Gpu,
    // Literals
    True,
    False,
    FloatLit(f64),
    IntLit(u64),
    StringLit(String),
    // Identifier
    Ident(String),
    // Assignment operators
    Eq,       // =
    PlusEq,   // +=
    MinusEq,  // -=
    StarEq,   // *=
    SlashEq,  // /=
    // Arithmetic (for key binding expressions)
    Plus,
    Minus,
    Star,
    Slash,
    Bang,     // ! (for boolean toggle)
    // Angle brackets (for parameterized types)
    Lt,       // <
    Gt,       // >
    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    Colon,
    Comma,
    DotDot,   // .. (for range)
    // End
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Pixel => write!(f, "pixel"),
            Token::Init => write!(f, "init"),
            Token::Kernel => write!(f, "kernel"),
            Token::Buffer => write!(f, "buffer"),
            Token::Constant => write!(f, "constant"),
            Token::Builtin => write!(f, "builtin"),
            Token::Var => write!(f, "var"),
            Token::Const => write!(f, "const"),
            Token::Pipeline => write!(f, "pipeline"),
            Token::Run => write!(f, "run"),
            Token::Display => write!(f, "display"),
            Token::Swap => write!(f, "swap"),
            Token::Loop => write!(f, "loop"),
            Token::Accumulate => write!(f, "accumulate"),
            Token::On => write!(f, "on"),
            Token::Keydown => write!(f, "keydown"),
            Token::Keypress => write!(f, "keypress"),
            Token::Keyup => write!(f, "keyup"),
            Token::Mousedown => write!(f, "mousedown"),
            Token::Click => write!(f, "click"),
            Token::Mouseup => write!(f, "mouseup"),
            Token::With => write!(f, "with"),
            Token::Include => write!(f, "include"),
            Token::Settings => write!(f, "settings"),
            Token::Title => write!(f, "title"),
            Token::Range => write!(f, "range"),
            Token::Gpu => write!(f, "gpu"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::FloatLit(v) => write!(f, "{v}"),
            Token::IntLit(v) => write!(f, "{v}"),
            Token::StringLit(s) => write!(f, "\"{s}\""),
            Token::Ident(s) => write!(f, "{s}"),
            Token::Eq => write!(f, "="),
            Token::PlusEq => write!(f, "+="),
            Token::MinusEq => write!(f, "-="),
            Token::StarEq => write!(f, "*="),
            Token::SlashEq => write!(f, "/="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Bang => write!(f, "!"),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::DotDot => write!(f, ".."),
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
