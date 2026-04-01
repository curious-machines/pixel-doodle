use std::fmt;

use super::span::Span;

#[derive(Debug)]
pub enum PdcError {
    Lex { span: Span, message: String },
    Parse { span: Span, message: String },
    Type { span: Span, message: String },
    Codegen { message: String },
}

impl PdcError {
    pub fn format(&self, source: &str) -> String {
        match self {
            PdcError::Lex { span, message } => {
                let (line, col) = span.line_col(source);
                format!("Lex error at {line}:{col}: {message}")
            }
            PdcError::Parse { span, message } => {
                let (line, col) = span.line_col(source);
                format!("Parse error at {line}:{col}: {message}")
            }
            PdcError::Type { span, message } => {
                let (line, col) = span.line_col(source);
                format!("Type error at {line}:{col}: {message}")
            }
            PdcError::Codegen { message } => {
                format!("Codegen error: {message}")
            }
        }
    }
}

impl fmt::Display for PdcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PdcError::Lex { message, .. } => write!(f, "lex error: {message}"),
            PdcError::Parse { message, .. } => write!(f, "parse error: {message}"),
            PdcError::Type { message, .. } => write!(f, "type error: {message}"),
            PdcError::Codegen { message } => write!(f, "codegen error: {message}"),
        }
    }
}
