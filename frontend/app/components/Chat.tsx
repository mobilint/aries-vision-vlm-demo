import { Grid2, Slider, Typography } from "@mui/material";
import { useRef } from "react";
import { LLMClient, LLMState } from "./type";
import Dialog from "./Dialog";
import { getLanguageTexts } from "../settings";

export default function Chat({
  client,
  language,
  isAutoMode,
  detectionThreshold,
  setDetectionThreshold,
}: {
  client: LLMClient,
  language: string,
  isAutoMode: boolean,
  detectionThreshold: number,
  setDetectionThreshold: (value: number) => void,
}) {
  const scrollGridRef = useRef<HTMLDivElement | null>(null);
  const texts = getLanguageTexts(language);
  const isWaitingForDetection = client.state == LLMState.IDLE;
  const isWaitingWithoutDialog = isWaitingForDetection && client.dialog.length == 0;
  const isWaitingAfterDialog = isWaitingForDetection && client.dialog.length > 0;
  const waitingText = isAutoMode
    ? texts.statusWaitingForDetection
    : texts.statusWaitingForManualSelection;

  return (
    <Grid2
      container
      direction="column"
      alignItems="center"
      size="grow"
      sx={{
        padding: "28px",
      }}
    >
      <Grid2
        container
        alignItems="center"
        columnSpacing="20px"
        sx={{
          width: "100%",
          maxWidth: "880px",
          padding: "4px 6px 0",
        }}
      >
        <Typography
          sx={{
            color: "#C8D1DC",
            fontSize: "16px",
            fontWeight: 500,
            whiteSpace: "nowrap",
          }}
        >
          {texts.detectionThresholdLabel}
        </Typography>
        <Grid2 size="grow" sx={{ minWidth: 0 }}>
          <Slider
            value={detectionThreshold}
            min={0}
            max={1}
            step={0.01}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => value.toFixed(2)}
            onChange={(_, value) => setDetectionThreshold(value as number)}
            sx={{
              color: "#1E88FF",
              "& .MuiSlider-thumb": {
                width: 16,
                height: 16,
              },
            }}
          />
        </Grid2>
        <Typography
          sx={{
            color: "#FFFFFF",
            fontSize: "16px",
            fontWeight: 600,
            minWidth: "44px",
            textAlign: "right",
          }}
        >
          {detectionThreshold.toFixed(2)}
        </Typography>
      </Grid2>
      <Grid2
        container
        size="grow"
        direction="column"
        wrap="nowrap"
        justifyContent={client.dialog.length == 0 ? "center" : undefined}
        alignItems="stretch"
        rowSpacing="44px"
        sx={{
          width: "100%",
          maxWidth: "880px",
          overflowY: "scroll",
          margin: "50px 0px",
        }}
        ref={scrollGridRef}
      >
        {isWaitingWithoutDialog &&
          <Grid2 container justifyContent="center" alignItems="center" sx={{ minHeight: "100%" }}>
            <Typography
              sx={{
                color: "#C8D1DC",
                fontSize: "28px",
                fontWeight: 500,
                lineHeight: "140%",
                textAlign: "center",
                letterSpacing: "-0.2px",
              }}
            >
              {waitingText}
            </Typography>
          </Grid2>
        }
        <Dialog
          client={client}
          language={language}
          scrollGridRef={scrollGridRef}
        />
      </Grid2>
      {isWaitingAfterDialog &&
        <Grid2
          container
          justifyContent="center"
          alignItems="center"
          sx={{
            width: "100%",
            maxWidth: "880px",
            paddingTop: "8px",
          }}
        >
          <Typography
            sx={{
              color: "#C8D1DC",
              fontSize: "22px",
              fontWeight: 500,
              lineHeight: "140%",
              textAlign: "center",
              letterSpacing: "-0.2px",
            }}
          >
            {waitingText}
          </Typography>
        </Grid2>
      }
    </Grid2>
  )
}
