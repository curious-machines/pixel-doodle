/// Byte-offset span in source code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Compute (line, col) from byte offset into source. 1-based.
    pub fn line_col(&self, source: &str) -> (u32, u32) {
        let mut line = 1u32;
        let mut col = 1u32;
        for (i, ch) in source.char_indices() {
            if i as u32 >= self.start {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }
}

/// AST node wrapped with source location and unique ID.
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
    pub id: u32,
}

/// Allocates unique IDs for AST nodes.
pub struct IdAlloc {
    next: u32,
}

impl IdAlloc {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    pub fn next(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    pub fn spanned<T>(&mut self, node: T, span: Span) -> Spanned<T> {
        Spanned {
            node,
            span,
            id: self.next(),
        }
    }
}
