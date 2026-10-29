import { CircularProgress, Typography } from '@mui/material';
import Grid2 from "@mui/material/Grid2"
import { Component, Fragment, ReactNode } from 'react';
import ReactMarkdown from "react-markdown"
import rehypeHighlight from 'rehype-highlight'
import { LLMClient } from './type';
import ModelIcon from './ModelIcon';

// Catches errors thrown from within the ReactMarkdown pipeline (e.g. a
// highlight parser blowing the stack on a pathological partial
// buffer) and falls back to a plain-text render of the same source.
class MarkdownErrorBoundary extends Component<
  { resetKey: string; fallback: ReactNode; children: ReactNode },
  { errored: boolean; lastResetKey: string }
> {
  constructor(props: { resetKey: string; fallback: ReactNode; children: ReactNode }) {
    super(props);
    this.state = { errored: false, lastResetKey: props.resetKey };
  }
  static getDerivedStateFromError() {
    return { errored: true };
  }
  static getDerivedStateFromProps(
    props: { resetKey: string },
    state: { errored: boolean; lastResetKey: string },
  ) {
    if (props.resetKey !== state.lastResetKey) {
      return { errored: false, lastResetKey: props.resetKey };
    }
    return null;
  }
  render() {
    return this.state.errored ? this.props.fallback : this.props.children;
  }
}

const plainTextFallbackStyle: React.CSSProperties = {
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  fontFamily: "inherit",
  fontSize: "inherit",
  lineHeight: "inherit",
  margin: 0,
};

// Latin keywords match on word boundaries; CJK keywords cannot use \b
// (JavaScript \w does not include Hangul/kanji/kana, so \b never fires
// around them) and instead match as substrings, which also covers
// conjugated forms such as 쓰러져/쓰러진 or 倒れて. Longer variants come
// before their substrings (手枪 before 枪) so the full word is bolded.
const WARNING_KEYWORDS_LATIN = [
  "weapon", "handgun", "shotgun", "gun", "knife", "rifle", "pistol", "firearm",
  "fallen", "falling", "falls", "fall", "fell", "collapsed", "lying",
  "urgent", "urgently", "urgency", "emergency", "immediately", "immediate",
  "assistance", "injured", "unconscious", "unsafe",
];
const WARNING_KEYWORDS_CJK = [
  "무기", "흉기", "권총", "소총", "총기", "총", "칼",
  "낙상", "쓰러", "넘어져", "넘어진", "긴급", "응급", "즉시", "부상", "도움",
  "武器", "拳銃", "銃", "ナイフ", "転倒", "倒れ", "緊急", "即時", "負傷",
  "手枪", "枪", "刀", "跌倒", "摔倒", "倒地", "紧急", "立即", "受伤",
];

const VLM_WARNING_KEYWORD_REGEX = new RegExp(
  `\\b(?:${WARNING_KEYWORDS_LATIN.join("|")})\\b|(?:${WARNING_KEYWORDS_CJK.join("|")})`,
  "gi",
);
const VLM_WARNING_KEYWORD_EXACT_REGEX = new RegExp(
  `^(?:${WARNING_KEYWORDS_LATIN.join("|")}|${WARNING_KEYWORDS_CJK.join("|")})$`,
  "i",
);

function emphasizePlainMarkdownText(text: string): string {
  return text.replace(VLM_WARNING_KEYWORD_REGEX, (match, offset: number, source: string) => {
    const before = source.slice(0, offset);
    const after = source.slice(offset + match.length);

    // Avoid turning an existing **keyword** into ****keyword****.
    if (before.endsWith("**") && after.startsWith("**")) {
      return match;
    }

    return `**${match}**`;
  });
}

// Keep code and math byte-for-byte intact. They are especially likely to be
// incomplete while a response is streaming.
function emphasizeWarningKeywords(text: string): string {
  const protectedPart = /(`{1,3}[\s\S]*?`{1,3}|\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])/g;
  let result = "";
  let lastIndex = 0;

  for (const match of text.matchAll(protectedPart)) {
    const index = match.index ?? 0;
    result += emphasizePlainMarkdownText(text.slice(lastIndex, index));
    result += match[0];
    lastIndex = index + match[0].length;
  }

  return result + emphasizePlainMarkdownText(text.slice(lastIndex));
}

function isWarningKeywordNode(children: React.ReactNode): boolean {
  const text = Array.isArray(children) ? children.join("") : String(children ?? "");
  return VLM_WARNING_KEYWORD_EXACT_REGEX.test(text.trim());
}

const markdownComponents = {
  strong({ children }: { children?: React.ReactNode }) {
    if (!isWarningKeywordNode(children)) {
      return <strong>{children}</strong>;
    }

    return (
      <strong style={{ color: "#FF4D4F", fontWeight: 800, textShadow: "0 0 12px rgba(255,77,79,0.28)" }}>
        {children}
      </strong>
    );
  },
};

export default function Answer({
  client,
  answer,
  isAnswering,
  isReasoningModel,
}: {
  client: LLMClient,
  answer: string | null,
  isAnswering: boolean,
  isReasoningModel: boolean,
}) {
  const thought_and_answer = !!answer && (isReasoningModel ? answer.split("</thought>") : ["", answer]);
  const thought = thought_and_answer && thought_and_answer[0];
  const real_answer = thought_and_answer && thought_and_answer[1];
  return (
    <Grid2
      container
      columnSpacing="20px"
      direction="row"
      wrap="nowrap"
      alignItems={thought_and_answer ? "flex-start" : "center"}
    >
      <Grid2
        container
        justifyContent="center"
        alignItems="center"
        style={{
          width: "38px",
          height: "38px",
          borderRadius: "55px",
          backgroundColor: "#FFFFFF",
          border: "1px solid #AAB8C2",
        }}
      >
        <ModelIcon
          model_id={client.model_id}
          width="22px"
        />
      </Grid2>
      <Grid2
        container
        size="grow"
        alignItems={thought_and_answer ? "flex-start" : "center"}
        sx={{
          fontFamily: "Pretendard",
          color: "white",
          fontSize: "21px",
          lineHeight: "160%",
          letterSpacing: "-0.3px",
          "& pre, & code": { fontFamily: "CascadiaCode" },
        }}
      >
      {thought_and_answer ?
        <Fragment>
        {thought &&
          <Grid2
            container
            direction="column"
            alignItems="flex-start"
            sx={{
              color: "#7C7C7E",
              "& > *:first-of-type": { marginTop: 0 },
              "& > *:last-of-type": { marginBottom: 0 },
            }}
          >
            {(() => {
              const source = (thought ?? "") + (isAnswering && !!answer == false ? " ..." : "");
              return (
                <MarkdownErrorBoundary
                  resetKey={source}
                  fallback={<pre style={plainTextFallbackStyle}>{source}</pre>}
                >
                  <ReactMarkdown
                    rehypePlugins={isAnswering ? [] : [rehypeHighlight]}
                    components={markdownComponents}
                  >
                    {emphasizeWarningKeywords(source)}
                  </ReactMarkdown>
                </MarkdownErrorBoundary>
              );
            })()}
          </Grid2>
        }{real_answer &&
          <Grid2
            container
            direction="column"
            alignItems="flex-start"
            sx={{
              "& > *:first-of-type": { marginTop: 0 },
              "& > *:last-of-type": { marginBottom: 0 },
            }}
          >
            {(() => {
              const source = (real_answer ?? "") + (isAnswering ? " ..." : "");
              return (
                <MarkdownErrorBoundary
                  resetKey={source}
                  fallback={<pre style={plainTextFallbackStyle}>{source}</pre>}
                >
                  <ReactMarkdown
                    rehypePlugins={isAnswering ? [] : [rehypeHighlight]}
                    components={markdownComponents}
                  >
                    {emphasizeWarningKeywords(source)}
                  </ReactMarkdown>
                </MarkdownErrorBoundary>
              );
            })()}
          </Grid2>
        }
        </Fragment> :
      isAnswering ?
        <Fragment>
          <CircularProgress size={38} />
        {client.tasksNum > 0 &&
          <Typography variant='caption'>
            Waiting for available device... ({client.tasksNum} {client.tasksNum == 1 ? "task" : "tasks"} waiting)
          </Typography>
        }
        </Fragment> :
        <Typography variant='caption' sx={{color: "#7C7C7E"}}>[Aborted]</Typography>
      }
      </Grid2>
    </Grid2>
  );
}
