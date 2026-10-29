import { Grid2 } from "@mui/material";
import { Fragment, MutableRefObject, useEffect, useRef } from "react";
import Answer from "./Answer";
import { LLMClient, LLMState } from "./type";
import { getLanguageTexts } from "../settings";

export default function Dialog({
  client,
  language,
  scrollGridRef,
}: {
  client: LLMClient,
  language: string,
  scrollGridRef: MutableRefObject<HTMLDivElement | null>,
}) {
  const isReasoningModel = [
    "LGAI-EXAONE/EXAONE-Deep-2.4B",
  ].includes(client.model_id);
  const texts = getLanguageTexts(language);

  const bottomDivRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = (smooth = false) => {
    bottomDivRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "end", inline: "end" })
  }

  useEffect(() => {
    if (scrollGridRef.current != null) {
      const diff = scrollGridRef.current.scrollHeight - scrollGridRef.current.offsetHeight - scrollGridRef.current.scrollTop;
      if (-100 < diff && diff < 100) {
        // Streaming updates arrive frequently. Smooth-scrolling every update
        // queues animations and can amplify recursive layout work in browsers.
        scrollToBottom(false);
      }
    }
  }, [client.recentAnswer])

  useEffect(() => {
    scrollToBottom(true);
  }, [client.dialog.length])

  return (
    <Fragment>
      {client.dialog.map((qna, index) =>
        <Fragment key={`${index}`}>
          <Grid2
            container
            direction="column"
            alignItems="flex-end"
            rowSpacing={"17px"}
          >
          </Grid2>
          {!(client.state != LLMState.IDLE && index == client.dialog.length - 1) &&
            <Answer
              client={client}
              answer={qna.answer}
              isAnswering={false}
              isReasoningModel={isReasoningModel}
            />
          }
        </Fragment>
      )}
      {client.state != LLMState.IDLE &&
        <Answer
          client={client}
          answer={client.recentAnswer}
          isAnswering={true}
          isReasoningModel={isReasoningModel}
        />
      }
      <div ref={bottomDivRef}></div>
    </Fragment>
  );
}
