//! Incremental parser for `text/event-stream` bodies.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Feeds the next chunk of the body and returns the events it completes.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut consumed = 0;
        while let Some(offset) = self.buffer[consumed..].iter().position(|byte| *byte == b'\n') {
            let end = consumed + offset;
            let line = String::from_utf8_lossy(&self.buffer[consumed..end])
                .trim_end_matches('\r')
                .to_owned();
            consumed = end + 1;
            events.extend(self.line(&line));
        }
        self.buffer.drain(..consumed);
        events
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            let event = self.event.take();
            return (!self.data.is_empty()).then(|| SseEvent {
                event,
                data: std::mem::take(&mut self.data).join("\n"),
            });
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events_split_across_chunks() {
        let mut parser = SseParser::default();
        assert_eq!(parser.push(b"event: a\r\ndata: {\"x\""), vec![]);
        assert_eq!(
            parser.push(b":1}\r\n\r\n: comment\ndata: one\ndata: two\n\n"),
            vec![
                SseEvent {
                    event: Some("a".into()),
                    data: "{\"x\":1}".into()
                },
                SseEvent {
                    event: None,
                    data: "one\ntwo".into()
                },
            ]
        );
    }
}
