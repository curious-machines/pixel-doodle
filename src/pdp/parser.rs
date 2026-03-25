use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;

use super::ast::*;
use super::token::{Spanned, Token};

#[derive(Debug)]
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

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    /// Directory of the file being parsed, for resolving includes.
    base_dir: PathBuf,
    /// Canonicalized paths already included, to prevent circular includes.
    included: Rc<RefCell<HashSet<PathBuf>>>,
}

impl Parser {
    #[cfg(test)]
    pub fn new(tokens: Vec<Spanned>) -> Self {
        Self {
            tokens,
            pos: 0,
            base_dir: PathBuf::from("."),
            included: Rc::new(RefCell::new(HashSet::new())),
        }
    }

    pub fn new_with_context(
        tokens: Vec<Spanned>,
        base_dir: PathBuf,
        included: Rc<RefCell<HashSet<PathBuf>>>,
    ) -> Self {
        Self {
            tokens,
            pos: 0,
            base_dir,
            included,
        }
    }

    // ── Token navigation ──

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn span(&self) -> Span {
        Span {
            line: self.tokens[self.pos].line,
            col: self.tokens[self.pos].col,
        }
    }

    fn advance(&mut self) -> &Spanned {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected: &Token) -> Result<&Spanned, ParseError> {
        if self.peek() == expected {
            Ok(self.advance())
        } else {
            Err(self.error(format!("expected '{expected}', got '{}'", self.peek())))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(self.error(format!("expected identifier, got '{other}'"))),
        }
    }

    fn expect_string(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            Token::StringLit(s) => {
                self.advance();
                Ok(s)
            }
            other => Err(self.error(format!("expected string literal, got '{other}'"))),
        }
    }

    fn error(&self, message: String) -> ParseError {
        let sp = &self.tokens[self.pos];
        ParseError {
            line: sp.line,
            col: sp.col,
            message,
        }
    }

    fn at(&self, tok: &Token) -> bool {
        self.peek() == tok
    }

    // ── Literal parsing ──

    fn parse_number_literal(&mut self) -> Result<f64, ParseError> {
        let negative = if self.at(&Token::Minus) {
            self.advance();
            true
        } else {
            false
        };
        let val = match self.peek().clone() {
            Token::FloatLit(v) => {
                self.advance();
                v
            }
            Token::IntLit(v) => {
                self.advance();
                v as f64
            }
            other => return Err(self.error(format!("expected number, got '{other}'"))),
        };
        Ok(if negative { -val } else { val })
    }

    fn parse_literal(&mut self) -> Result<Literal, ParseError> {
        match self.peek().clone() {
            Token::True => {
                self.advance();
                Ok(Literal::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Literal::Bool(false))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(Literal::Str(s))
            }
            Token::FloatLit(_) | Token::IntLit(_) | Token::Minus => {
                let v = self.parse_number_literal()?;
                Ok(Literal::Float(v))
            }
            other => Err(self.error(format!("expected literal, got '{other}'"))),
        }
    }

    // ── Named arguments: (name: value, ...) ──

    fn parse_named_args(&mut self) -> Result<Vec<NamedArg>, ParseError> {
        self.expect(&Token::LParen)?;
        let mut args = Vec::new();
        while !self.at(&Token::RParen) {
            let span = self.span();
            let name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let value = self.parse_literal()?;
            args.push(NamedArg { name, value, span });
            if !self.at(&Token::RParen) {
                self.expect(&Token::Comma)?;
            }
        }
        self.expect(&Token::RParen)?;
        Ok(args)
    }

    // ── Top-level parsing ──

    pub fn parse_config(&mut self) -> Result<Config, ParseError> {
        let mut config = Config {
            title: None,
            variables: Vec::new(),
            settings: Settings::default(),
            key_bindings: Vec::new(),
            pipelines: Vec::new(),
        };

        while !self.at(&Token::Eof) {
            match self.peek().clone() {
                Token::Pixel | Token::Sim | Token::Init => {
                    return Err(self.error(
                        "kernel declarations must be inside a pipeline block".into(),
                    ));
                }
                Token::Buffer => {
                    return Err(self.error(
                        "buffer declarations must be inside a pipeline block".into(),
                    ));
                }
                Token::Title => {
                    self.advance();
                    self.expect(&Token::Eq)?;
                    config.title = Some(self.expect_string()?);
                }
                Token::Settings => {
                    config.settings = self.parse_settings()?;
                }
                Token::Pipeline => {
                    config.pipelines.push(self.parse_pipeline()?);
                }
                Token::On => {
                    config.key_bindings.push(self.parse_key_binding()?);
                }
                Token::Include => {
                    let included = self.parse_include()?;
                    if config.title.is_none() {
                        config.title = included.title;
                    }
                    config.variables.extend(included.variables);
                    config.key_bindings.extend(included.key_bindings);
                    config.settings.entries.extend(included.settings.entries);
                }
                Token::Ident(_) => {
                    // Could be a variable declaration: `name = value` or `name: range(...) = value`
                    config.variables.push(self.parse_var_decl()?);
                }
                other => {
                    return Err(self.error(format!(
                        "unexpected token '{other}' at top level"
                    )));
                }
            }
        }

        Ok(config)
    }

    // ── Kernel declarations ──

    fn parse_kernel_decl(&mut self) -> Result<KernelDecl, ParseError> {
        let span = self.span();
        let kind = match self.peek() {
            Token::Pixel => {
                self.advance();
                KernelKind::Pixel
            }
            Token::Sim => {
                self.advance();
                KernelKind::Sim
            }
            Token::Init => {
                self.advance();
                KernelKind::Init
            }
            _ => return Err(self.error("expected 'pixel', 'sim', or 'init'".into())),
        };
        self.expect(&Token::Kernel)?;

        // Either: "path.pd" (unnamed) or name = "path.pd" (named)
        match self.peek().clone() {
            Token::StringLit(path) => {
                self.advance();
                // Derive name from filename
                let name = derive_kernel_name(&path);
                Ok(KernelDecl {
                    kind,
                    name,
                    path,
                    span,
                })
            }
            Token::Ident(name) => {
                self.advance();
                self.expect(&Token::Eq)?;
                let path = self.expect_string()?;
                Ok(KernelDecl {
                    kind,
                    name,
                    path,
                    span,
                })
            }
            other => Err(self.error(format!(
                "expected kernel path or name, got '{other}'"
            ))),
        }
    }

    // ── Buffer declarations ──

    fn parse_buffer_decl(&mut self) -> Result<BufferDecl, ParseError> {
        let span = self.span();
        self.expect(&Token::Buffer)?;
        let name = self.expect_ident()?;

        // Optional GPU type annotation: `: gpu(vec2f)`
        let gpu_type = if self.at(&Token::Colon) {
            self.advance();
            self.expect(&Token::Gpu)?;
            self.expect(&Token::LParen)?;
            let type_name = self.expect_ident()?;
            let gt = match type_name.as_str() {
                "f32" => GpuElementType::F32,
                "vec2f" => GpuElementType::Vec2f,
                "vec3f" => GpuElementType::Vec3f,
                "vec4f" => GpuElementType::Vec4f,
                "i32" => GpuElementType::I32,
                "u32" => GpuElementType::U32,
                other => {
                    return Err(self.error(format!(
                        "unknown GPU element type '{other}'. Expected: f32, vec2f, vec3f, vec4f, i32, u32"
                    )))
                }
            };
            self.expect(&Token::RParen)?;
            Some(gt)
        } else {
            None
        };

        self.expect(&Token::Eq)?;

        let init = if self.at(&Token::Constant) {
            self.advance();
            self.expect(&Token::LParen)?;
            let val = self.parse_number_literal()?;
            self.expect(&Token::RParen)?;
            BufferInit::Constant(val)
        } else {
            // init_kernel_name(args...)
            let kernel_name = self.expect_ident()?;
            let args = if self.at(&Token::LParen) {
                self.parse_named_args()?
            } else {
                Vec::new()
            };
            BufferInit::InitKernel { kernel_name, args }
        };

        Ok(BufferDecl { name, gpu_type, init, span })
    }

    // ── Variable declarations ──

    fn parse_var_decl(&mut self) -> Result<VarDecl, ParseError> {
        let span = self.span();
        let name = self.expect_ident()?;

        // Optional range annotation: `: range(min..max)` or `: range(min..max, wrap: true)`
        let range = if self.at(&Token::Colon) {
            self.advance();
            self.expect(&Token::Range)?;
            self.expect(&Token::LParen)?;
            let min = self.parse_number_literal()?;
            self.expect(&Token::DotDot)?;
            let max = self.parse_number_literal()?;
            let mut wrap = false;
            if self.at(&Token::Comma) {
                self.advance();
                // expect "wrap: true" or "wrap: false"
                let key = self.expect_ident()?;
                if key != "wrap" {
                    return Err(self.error(format!("expected 'wrap', got '{key}'")));
                }
                self.expect(&Token::Colon)?;
                wrap = match self.peek() {
                    Token::True => {
                        self.advance();
                        true
                    }
                    Token::False => {
                        self.advance();
                        false
                    }
                    other => {
                        return Err(
                            self.error(format!("expected 'true' or 'false', got '{other}'"))
                        )
                    }
                };
            }
            self.expect(&Token::RParen)?;
            Some(RangeSpec { min, max, wrap })
        } else {
            None
        };

        self.expect(&Token::Eq)?;
        let default = self.parse_literal()?;

        Ok(VarDecl {
            name,
            range,
            default,
            span,
        })
    }

    // ── Settings block ──

    fn parse_settings(&mut self) -> Result<Settings, ParseError> {
        self.expect(&Token::Settings)?;
        self.expect(&Token::LBrace)?;
        let mut entries = Vec::new();
        while !self.at(&Token::RBrace) {
            let span = self.span();
            let key = self.expect_ident()?;
            self.expect(&Token::Eq)?;
            let value = self.parse_literal()?;
            entries.push(SettingsEntry { key, value, span });
        }
        self.expect(&Token::RBrace)?;
        Ok(Settings { entries })
    }

    // ── Key bindings ──

    fn parse_key_binding(&mut self) -> Result<KeyBinding, ParseError> {
        let span = self.span();
        self.expect(&Token::On)?;
        self.expect(&Token::Key)?;
        self.expect(&Token::LParen)?;
        let key_name = match self.peek().clone() {
            Token::Ident(s) => { self.advance(); s }
            Token::IntLit(n) if n <= 9 => { self.advance(); n.to_string() }
            Token::IntLit(n) => {
                return Err(self.error(format!(
                    "key name must be a single digit (0-9), got '{n}'"
                )));
            }
            other => {
                return Err(self.error(format!("expected key name, got '{other}'")));
            }
        };
        self.expect(&Token::RParen)?;

        let actions = if self.at(&Token::LBrace) {
            self.advance();
            let mut actions = Vec::new();
            while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
                actions.push(self.parse_action()?);
            }
            self.expect(&Token::RBrace)?;
            actions
        } else {
            vec![self.parse_action()?]
        };

        Ok(KeyBinding {
            key_name,
            actions,
            span,
        })
    }

    fn parse_action(&mut self) -> Result<Action, ParseError> {
        // Forms:
        //   variable = !variable          (toggle)
        //   variable = literal            (direct assign)
        //   variable += literal           (compound assign)
        //   variable -= literal
        //   variable *= literal
        //   variable /= literal
        //   variable = variable + literal (expanded form)
        //   variable = variable - literal
        //   variable = variable * literal
        //   variable = variable / literal
        let target = self.expect_ident()?;

        // Bare keyword actions (no assignment operator follows)
        if target == "quit" {
            return Ok(Action::Quit);
        }

        match self.peek().clone() {
            Token::PlusEq => {
                self.advance();
                let val = self.parse_number_literal()?;
                Ok(Action::CompoundAssign {
                    target,
                    op: CompoundOp::Add,
                    value: val,
                })
            }
            Token::MinusEq => {
                self.advance();
                let val = self.parse_number_literal()?;
                Ok(Action::CompoundAssign {
                    target,
                    op: CompoundOp::Sub,
                    value: val,
                })
            }
            Token::StarEq => {
                self.advance();
                let val = self.parse_number_literal()?;
                Ok(Action::CompoundAssign {
                    target,
                    op: CompoundOp::Mul,
                    value: val,
                })
            }
            Token::SlashEq => {
                self.advance();
                let val = self.parse_number_literal()?;
                Ok(Action::CompoundAssign {
                    target,
                    op: CompoundOp::Div,
                    value: val,
                })
            }
            Token::Eq => {
                self.advance();
                if self.at(&Token::Bang) {
                    // `!variable` (toggle)
                    self.advance();
                    let var = self.expect_ident()?;
                    Ok(Action::Toggle(var))
                } else if matches!(self.peek(), Token::FloatLit(_) | Token::IntLit(_) | Token::Minus) {
                    // literal (direct assign)
                    let val = self.parse_number_literal()?;
                    Ok(Action::Assign { target, value: val })
                } else {
                    // variable op literal (expanded form)
                    let _rhs_var = self.expect_ident()?;
                    let op = match self.peek() {
                        Token::Plus => CompoundOp::Add,
                        Token::Minus => CompoundOp::Sub,
                        Token::Star => CompoundOp::Mul,
                        Token::Slash => CompoundOp::Div,
                        other => {
                            return Err(self.error(format!(
                                "expected operator (+, -, *, /), got '{other}'"
                            )))
                        }
                    };
                    self.advance();
                    let val = self.parse_number_literal()?;
                    Ok(Action::BinAssign {
                        target,
                        op,
                        value: val,
                    })
                }
            }
            other => Err(self.error(format!(
                "expected assignment operator, got '{other}'"
            ))),
        }
    }

    // ── Include ──

    fn parse_include(&mut self) -> Result<Config, ParseError> {
        self.expect(&Token::Include)?;
        let path_str = self.expect_string()?;

        let resolved = self.base_dir.join(&path_str);
        let canonical = resolved.canonicalize().map_err(|e| {
            self.error(format!("cannot resolve include path '{path_str}': {e}"))
        })?;

        // Dedup: if already included, return empty config
        if !self.included.borrow_mut().insert(canonical.clone()) {
            return Ok(Config {
                title: None,
                variables: Vec::new(),
                settings: Settings::default(),
                key_bindings: Vec::new(),
                pipelines: Vec::new(),
            });
        }

        let source = std::fs::read_to_string(&canonical).map_err(|e| {
            self.error(format!("cannot read include file '{path_str}': {e}"))
        })?;

        let tokens = super::lexer::lex(&source).map_err(|e| {
            self.error(format!("in included file '{path_str}': {e}"))
        })?;

        let sub_dir = canonical.parent().unwrap_or(&self.base_dir).to_path_buf();
        let mut sub_parser =
            Parser::new_with_context(tokens, sub_dir, Rc::clone(&self.included));

        sub_parser.parse_include_file(&path_str)
    }

    fn parse_include_file(&mut self, path_str: &str) -> Result<Config, ParseError> {
        let mut config = Config {
            title: None,
            variables: Vec::new(),
            settings: Settings::default(),
            key_bindings: Vec::new(),
            pipelines: Vec::new(),
        };

        while !self.at(&Token::Eof) {
            match self.peek().clone() {
                Token::Pipeline => {
                    return Err(self.error(format!(
                        "included file '{path_str}' must not contain pipeline blocks"
                    )));
                }
                Token::Pixel | Token::Sim | Token::Init => {
                    return Err(self.error(format!(
                        "included file '{path_str}' must not contain kernel declarations"
                    )));
                }
                Token::Buffer => {
                    return Err(self.error(format!(
                        "included file '{path_str}' must not contain buffer declarations"
                    )));
                }
                Token::Title => {
                    self.advance();
                    self.expect(&Token::Eq)?;
                    config.title = Some(self.expect_string()?);
                }
                Token::Settings => {
                    config.settings = self.parse_settings()?;
                }
                Token::On => {
                    config.key_bindings.push(self.parse_key_binding()?);
                }
                Token::Include => {
                    let included = self.parse_include()?;
                    if config.title.is_none() {
                        config.title = included.title;
                    }
                    config.variables.extend(included.variables);
                    config.key_bindings.extend(included.key_bindings);
                    config.settings.entries.extend(included.settings.entries);
                }
                Token::Ident(_) => {
                    config.variables.push(self.parse_var_decl()?);
                }
                other => {
                    return Err(self.error(format!(
                        "unexpected token '{other}' in included file '{path_str}'"
                    )));
                }
            }
        }

        Ok(config)
    }

    // ── Pipeline ──

    fn parse_pipeline(&mut self) -> Result<Pipeline, ParseError> {
        let span = self.span();
        self.expect(&Token::Pipeline)?;

        // Optional pipeline name: `pipeline cpu {` or `pipeline gpu {` or `pipeline {`
        let name = if self.at(&Token::LBrace) {
            None
        } else if let Token::Ident(name) = self.peek().clone() {
            self.advance();
            Some(name)
        } else if self.at(&Token::Gpu) {
            // 'gpu' is a keyword token but valid as a pipeline name
            self.advance();
            Some("gpu".to_string())
        } else {
            None
        };

        self.expect(&Token::LBrace)?;

        // Pipeline body can contain kernel/buffer declarations AND pipeline steps
        let mut kernels = Vec::new();
        let mut buffers = Vec::new();
        let mut steps = Vec::new();

        while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
            match self.peek().clone() {
                Token::Pixel | Token::Sim | Token::Init => {
                    kernels.push(self.parse_kernel_decl()?);
                }
                Token::Buffer => {
                    buffers.push(self.parse_buffer_decl()?);
                }
                _ => {
                    steps.push(self.parse_pipeline_step()?);
                }
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(Pipeline {
            name,
            kernels,
            buffers,
            steps,
            span,
        })
    }

    fn parse_pipeline_steps(&mut self) -> Result<Vec<PipelineStep>, ParseError> {
        let mut steps = Vec::new();
        while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
            steps.push(self.parse_pipeline_step()?);
        }
        Ok(steps)
    }

    fn parse_pipeline_step(&mut self) -> Result<PipelineStep, ParseError> {
        match self.peek().clone() {
            Token::Run => self.parse_run_step(false),
            Token::Display => self.parse_run_step(true),
            Token::Swap => self.parse_swap_step(),
            Token::Loop => self.parse_loop_step(),
            Token::Accumulate => self.parse_accumulate_step(),
            Token::On => self.parse_on_event_step(),
            Token::Ident(_) => {
                // Tuple assignment: `a, b = run kernel ...` or `a = run kernel ...`
                self.parse_assignment_step()
            }
            other => Err(self.error(format!(
                "expected pipeline step (run, display, swap, loop, accumulate, on), got '{other}'"
            ))),
        }
    }

    /// Parse `run kernel(args...) { bindings }` or `display kernel(args...) { bindings }`
    fn parse_run_step(&mut self, is_display: bool) -> Result<PipelineStep, ParseError> {
        let span = self.span();
        self.advance(); // consume 'run' or 'display'

        let kernel_name = self.expect_ident()?;
        let args = if self.at(&Token::LParen) {
            self.parse_named_args()?
        } else {
            Vec::new()
        };
        let input_bindings = if self.at(&Token::LBrace) {
            self.parse_buffer_bindings()?
        } else {
            Vec::new()
        };

        if is_display {
            Ok(PipelineStep::Display {
                outputs: Vec::new(),
                kernel_name,
                args,
                input_bindings,
                span,
            })
        } else {
            Ok(PipelineStep::Run {
                outputs: Vec::new(),
                kernel_name,
                args,
                input_bindings,
                span,
            })
        }
    }

    /// Parse `a, b = run|display kernel ...`
    fn parse_assignment_step(&mut self) -> Result<PipelineStep, ParseError> {
        let span = self.span();
        let mut outputs = Vec::new();
        outputs.push(self.expect_ident()?);

        // Collect comma-separated output names
        while self.at(&Token::Comma) {
            self.advance();
            outputs.push(self.expect_ident()?);
        }

        self.expect(&Token::Eq)?;

        let is_display = match self.peek() {
            Token::Run => false,
            Token::Display => true,
            other => {
                return Err(self.error(format!(
                    "expected 'run' or 'display' after '=', got '{other}'"
                )))
            }
        };
        self.advance();

        let kernel_name = self.expect_ident()?;
        let args = if self.at(&Token::LParen) {
            self.parse_named_args()?
        } else {
            Vec::new()
        };
        let input_bindings = if self.at(&Token::LBrace) {
            self.parse_buffer_bindings()?
        } else {
            Vec::new()
        };

        if is_display {
            Ok(PipelineStep::Display {
                outputs,
                kernel_name,
                args,
                input_bindings,
                span,
            })
        } else {
            Ok(PipelineStep::Run {
                outputs,
                kernel_name,
                args,
                input_bindings,
                span,
            })
        }
    }

    fn parse_buffer_bindings(&mut self) -> Result<Vec<BufferBinding>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut bindings = Vec::new();
        while !self.at(&Token::RBrace) {
            let span = self.span();
            let param_name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            // Check for `out` qualifier: `pixels: out pixels`
            let is_output = matches!(self.peek(), Token::Ident(name) if name == "out");
            if is_output {
                self.advance();
            }
            let buffer_name = self.expect_ident()?;
            bindings.push(BufferBinding {
                param_name,
                buffer_name,
                is_output,
                span,
            });
            if !self.at(&Token::RBrace) {
                self.expect(&Token::Comma)?;
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(bindings)
    }

    fn parse_swap_step(&mut self) -> Result<PipelineStep, ParseError> {
        let span = self.span();
        self.expect(&Token::Swap)?;

        let mut pairs = Vec::new();
        loop {
            let a = self.expect_ident()?;
            self.expect(&Token::SwapArrow)?;
            let b = self.expect_ident()?;
            pairs.push((a, b));
            if self.at(&Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(PipelineStep::Swap { pairs, span })
    }

    fn parse_loop_step(&mut self) -> Result<PipelineStep, ParseError> {
        let span = self.span();
        self.expect(&Token::Loop)?;
        self.expect(&Token::LParen)?;

        // expect "iterations: N" or "iterations: var_name"
        let key = self.expect_ident()?;
        if key != "iterations" {
            return Err(self.error(format!("expected 'iterations', got '{key}'")));
        }
        self.expect(&Token::Colon)?;

        let iterations = match self.peek().clone() {
            Token::IntLit(n) => {
                self.advance();
                IterCount::Fixed(n as u32)
            }
            Token::Ident(name) => {
                self.advance();
                IterCount::Variable(name)
            }
            other => {
                return Err(self.error(format!(
                    "expected integer or variable name, got '{other}'"
                )))
            }
        };

        self.expect(&Token::RParen)?;
        self.expect(&Token::LBrace)?;
        let body = self.parse_pipeline_steps()?;
        self.expect(&Token::RBrace)?;

        Ok(PipelineStep::Loop {
            iterations,
            body,
            span,
        })
    }

    fn parse_accumulate_step(&mut self) -> Result<PipelineStep, ParseError> {
        let span = self.span();
        self.expect(&Token::Accumulate)?;
        self.expect(&Token::LParen)?;

        let key = self.expect_ident()?;
        if key != "samples" {
            return Err(self.error(format!("expected 'samples', got '{key}'")));
        }
        self.expect(&Token::Colon)?;

        let samples = match self.peek().clone() {
            Token::IntLit(n) => {
                self.advance();
                n as u32
            }
            other => {
                return Err(self.error(format!("expected integer, got '{other}'")))
            }
        };

        self.expect(&Token::RParen)?;
        self.expect(&Token::LBrace)?;
        let body = self.parse_pipeline_steps()?;
        self.expect(&Token::RBrace)?;

        Ok(PipelineStep::Accumulate {
            samples,
            body,
            span,
        })
    }

    fn parse_on_event_step(&mut self) -> Result<PipelineStep, ParseError> {
        let span = self.span();
        self.expect(&Token::On)?;

        match self.peek().clone() {
            Token::Click => {
                self.advance();
                // Parse options: (continuous: true)
                let mut continuous = false;
                if self.at(&Token::LParen) {
                    self.advance();
                    while !self.at(&Token::RParen) {
                        let key = self.expect_ident()?;
                        self.expect(&Token::Colon)?;
                        match key.as_str() {
                            "continuous" => {
                                continuous = match self.peek() {
                                    Token::True => {
                                        self.advance();
                                        true
                                    }
                                    Token::False => {
                                        self.advance();
                                        false
                                    }
                                    other => {
                                        return Err(self.error(format!(
                                            "expected 'true' or 'false', got '{other}'"
                                        )))
                                    }
                                };
                            }
                            _ => {
                                return Err(
                                    self.error(format!("unknown click option '{key}'"))
                                )
                            }
                        }
                        if !self.at(&Token::RParen) {
                            self.expect(&Token::Comma)?;
                        }
                    }
                    self.expect(&Token::RParen)?;
                }
                self.expect(&Token::LBrace)?;
                let body = self.parse_pipeline_steps()?;
                self.expect(&Token::RBrace)?;

                Ok(PipelineStep::OnClick {
                    continuous,
                    body,
                    span,
                })
            }
            other => Err(self.error(format!("expected 'click', got '{other}'"))),
        }
    }
}

/// Derive a kernel name from a file path by taking the base name without extension
/// and replacing non-alphanumeric chars with underscores.
fn derive_kernel_name(path: &str) -> String {
    let base = path
        .rsplit('/')
        .next()
        .unwrap_or(path)
        .rsplit('\\')
        .next()
        .unwrap_or(path);

    let name = if let Some(dot_pos) = base.rfind('.') {
        &base[..dot_pos]
    } else {
        base
    };

    let mut result = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            result.push(ch);
        } else {
            result.push('_');
        }
    }

    // Ensure it doesn't start with a digit
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert(0, '_');
    }

    if result.is_empty() {
        result = "unnamed".into();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdp::lexer::lex;

    fn parse_str(input: &str) -> Result<Config, ParseError> {
        let tokens = lex(input).map_err(|e| ParseError {
            line: 0,
            col: 0,
            message: e,
        })?;
        let mut parser = Parser::new(tokens);
        parser.parse_config()
    }

    #[test]
    fn parse_gradient() {
        let config = parse_str(
            r#"
            pipeline {
              pixel kernel "gradient.pd"
              display gradient
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.pipelines.len(), 1);
        assert_eq!(config.pipelines[0].kernels.len(), 1);
        assert_eq!(config.pipelines[0].kernels[0].kind, KernelKind::Pixel);
        assert_eq!(config.pipelines[0].kernels[0].name, "gradient");
        assert_eq!(config.pipelines[0].kernels[0].path, "gradient.pd");
        let steps = &config.pipelines[0].steps;
        assert_eq!(steps.len(), 1);
        assert!(matches!(&steps[0], PipelineStep::Display { kernel_name, .. } if kernel_name == "gradient"));
    }

    #[test]
    fn parse_mandelbrot_progressive() {
        let config = parse_str(
            r#"
            on key(left) center_x -= 0.1
            on key(right) center_x += 0.1
            on key(plus) zoom *= 1.1

            pipeline {
              pixel kernel "mandelbrot.pd"
              accumulate(samples: 256) {
                display mandelbrot
              }
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.pipelines[0].kernels.len(), 1);
        assert_eq!(config.key_bindings.len(), 3);
        let pipeline = config.pipelines.into_iter().next().unwrap();
        assert_eq!(pipeline.steps.len(), 1);
        assert!(matches!(&pipeline.steps[0], PipelineStep::Accumulate { samples: 256, .. }));
    }

    #[test]
    fn parse_gray_scott() {
        let config = parse_str(
            r#"
            title = "Gray-Scott"

            on key(space) paused = !paused
            on key(period) frame += 1

            pipeline {
              sim kernel "gray_scott.pd"
              init kernel init_u = "init/gray_scott_u.pd"
              init kernel init_v = "init/gray_scott_v.pd"

              buffer u = init_u()
              buffer v = init_v()
              buffer u_next = constant(0.0)
              buffer v_next = constant(0.0)

              on click(continuous: true) {
                v = run inject(value: 0.5, radius: 5)
              }
              loop(iterations: 8) {
                u_next, v_next = display gray_scott { u_in: u, v_in: v }
                swap u <-> u_next
                swap v <-> v_next
              }
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.title.as_deref(), Some("Gray-Scott"));
        assert_eq!(config.pipelines[0].kernels.len(), 3);
        assert_eq!(config.pipelines[0].buffers.len(), 4);
        assert!(matches!(&config.pipelines[0].buffers[0].init, BufferInit::InitKernel { kernel_name, .. } if kernel_name == "init_u"));
        assert!(matches!(&config.pipelines[0].buffers[2].init, BufferInit::Constant(v) if *v == 0.0));
        assert_eq!(config.key_bindings.len(), 2);

        let pipeline = config.pipelines.into_iter().next().unwrap();
        assert_eq!(pipeline.steps.len(), 2); // on click, loop
        assert!(matches!(&pipeline.steps[0], PipelineStep::OnClick { continuous: true, .. }));
        if let PipelineStep::Loop { body, .. } = &pipeline.steps[1] {
            assert_eq!(body.len(), 3); // display, swap, swap
        } else {
            panic!("expected loop step");
        }
    }

    #[test]
    fn parse_smoke() {
        let config = parse_str(
            r#"
            title = "Smoke Simulation"

            pipeline {
              sim kernel advect = "smoke/advect.pd"
              sim kernel divergence = "smoke/divergence.pd"
              sim kernel jacobi = "smoke/jacobi.pd"
              sim kernel project = "smoke/project.pd"

              buffer vx = constant(0.0)
              buffer vy = constant(0.0)
              buffer density = constant(0.0)
              buffer vx0 = constant(0.0)
              buffer vy0 = constant(0.0)
              buffer density0 = constant(0.0)
              buffer pressure = constant(0.0)
              buffer pressure_tmp = constant(0.0)
              buffer divergence = constant(0.0)

              swap vx <-> vx0, vy <-> vy0, density <-> density0
              vx, vy, density = run advect { vx_in: vx0, vy_in: vy0, den_in: density0 }
              divergence = run divergence { vx_in: vx, vy_in: vy }
              loop(iterations: 40) {
                pressure_tmp = run jacobi { div_in: divergence, p_in: pressure }
                swap pressure <-> pressure_tmp
              }
              vx0, vy0 = display project { p_in: pressure, vx_in: vx, vy_in: vy, den_in: density }
              swap vx <-> vx0, vy <-> vy0
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.pipelines[0].kernels.len(), 4);
        assert_eq!(config.pipelines[0].buffers.len(), 9);
        let pipeline = config.pipelines.into_iter().next().unwrap();
        assert_eq!(pipeline.steps.len(), 6); // swap, run, run, loop, display, swap
    }

    #[test]
    fn parse_game_of_life() {
        let config = parse_str(
            r#"
            iterations: range(1..10) = 1

            on key(space) paused = !paused
            on key(period) frame += 1
            on key(bracket_right) iterations += 1
            on key(bracket_left) iterations -= 1

            pipeline {
              sim kernel "game_of_life.pd"
              init kernel init_state = "init/random_binary.pd"

              buffer state = init_state(density: 0.3, seed: 42)
              buffer age = constant(0.0)
              buffer state_next = constant(0.0)
              buffer age_next = constant(0.0)

              on click(continuous: true) {
                state = run inject(value: 1.0, radius: 3)
                age = run inject(value: 0.0, radius: 3)
              }
              loop(iterations: iterations) {
                state_next, age_next = display game_of_life { state_in: state, age_in: age }
                swap state <-> state_next
                swap age <-> age_next
              }
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.variables.len(), 1);
        let var = &config.variables[0];
        assert_eq!(var.name, "iterations");
        assert!(var.range.is_some());
        let range = var.range.as_ref().unwrap();
        assert_eq!(range.min, 1.0);
        assert_eq!(range.max, 10.0);
        assert!(!range.wrap);

        assert_eq!(config.key_bindings.len(), 4);
    }

    #[test]
    fn parse_settings_block() {
        let config = parse_str(
            r#"
            settings {
              threads = 4
              backend = "cranelift"
              tile_height = 8
            }

            pipeline {
              pixel kernel "gradient.pd"
              display gradient
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.settings.entries.len(), 3);
        assert_eq!(config.settings.entries[0].key, "threads");
        assert_eq!(config.settings.entries[1].key, "backend");
    }

    #[test]
    fn parse_var_with_wrap() {
        let config = parse_str(
            r#"
            mode: range(0..3, wrap: true) = 0
            pipeline {
              pixel kernel "test.pd"
              display test
            }
            "#,
        )
        .unwrap();

        let var = &config.variables[0];
        assert!(var.range.as_ref().unwrap().wrap);
    }

    #[test]
    fn parse_named_pipelines() {
        let config = parse_str(
            r#"
            title = "Gray-Scott"

            on key(space) paused = !paused

            pipeline cpu {
              sim kernel "gray_scott.pd"
              buffer u = constant(1.0)
              buffer v = constant(0.0)

              loop(iterations: 8) {
                u, v = display gray_scott { u_in: u, v_in: v }
              }
            }

            pipeline gpu {
              sim kernel step = "gray_scott_step.wgsl"
              sim kernel vis = "gray_scott_vis.wgsl"
              buffer field = constant(0.0)
              buffer field_next = constant(0.0)

              loop(iterations: 8) {
                field_next = run step { field_in: field }
                swap field <-> field_next
              }
              display vis { field_in: field }
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.pipelines.len(), 2);
        assert_eq!(config.pipelines[0].name.as_deref(), Some("cpu"));
        assert_eq!(config.pipelines[1].name.as_deref(), Some("gpu"));
        // CPU pipeline has its own kernels and buffers
        assert_eq!(config.pipelines[0].kernels.len(), 1);
        assert_eq!(config.pipelines[0].buffers.len(), 2);
        // GPU pipeline has its own kernels and buffers
        assert_eq!(config.pipelines[1].kernels.len(), 2);
        assert_eq!(config.pipelines[1].buffers.len(), 2);
        // Shared key bindings at top level
        assert_eq!(config.key_bindings.len(), 1);
    }

    #[test]
    fn parse_key_block() {
        let config = parse_str(
            r#"
            on key(0) {
              center_x = 0.0
              center_y = 0.0
              zoom = 1.0
            }

            pipeline {
              pixel kernel "gradient.pd"
              display gradient
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.key_bindings.len(), 1);
        let kb = &config.key_bindings[0];
        assert_eq!(kb.key_name, "0");
        assert_eq!(kb.actions.len(), 3);
        assert!(matches!(&kb.actions[0], Action::Assign { target, value } if target == "center_x" && *value == 0.0));
        assert!(matches!(&kb.actions[1], Action::Assign { target, value } if target == "center_y" && *value == 0.0));
        assert!(matches!(&kb.actions[2], Action::Assign { target, value } if target == "zoom" && *value == 1.0));
    }

    #[test]
    fn parse_direct_assign() {
        let config = parse_str(
            r#"
            on key(0) zoom = 1.0
            on key(1) center_x = -0.5

            pipeline {
              pixel kernel "gradient.pd"
              display gradient
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.key_bindings.len(), 2);
        assert!(matches!(&config.key_bindings[0].actions[0], Action::Assign { target, value } if target == "zoom" && *value == 1.0));
        assert!(matches!(&config.key_bindings[1].actions[0], Action::Assign { target, value } if target == "center_x" && *value == -0.5));
    }

    #[test]
    fn derive_name_from_path() {
        assert_eq!(derive_kernel_name("gradient.pd"), "gradient");
        assert_eq!(derive_kernel_name("smoke/advect.pd"), "advect");
        assert_eq!(derive_kernel_name("my-kernel.pd"), "my_kernel");
        assert_eq!(derive_kernel_name("123bad.pd"), "_123bad");
    }
}
