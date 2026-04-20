import { Grid2, Box } from "@mui/material";
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

  const scrollToBottom = () => {
    bottomDivRef.current?.scrollIntoView({ behavior: "smooth", block: "end", inline: "end" })
  }

  useEffect(() => {
    if (scrollGridRef.current != null) {
      const diff = scrollGridRef.current.scrollHeight - scrollGridRef.current.offsetHeight - scrollGridRef.current.scrollTop;
      if (-100 < diff && diff < 100)
        scrollToBottom();
    }
  }, [client.recentAnswer])

  useEffect(() => {
    scrollToBottom();
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
          {index == 0 && client.image &&
            <Box
              component="img"
              src={client.image}
              alt={texts.imagePanelTitle}
              sx={{
                display: "block",
                width: "100%",
                maxWidth: "560px",
                maxHeight: "560px",
                borderRadius: "20px",
                borderBottomRightRadius: "5px",
                objectFit: "contain",
                backgroundColor: "#0F1114",
              }}
            />
          }
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
