import { Box, Grid2, Slider, Tooltip, Typography } from "@mui/material";
import { useRef } from "react";
import { LLMClient, LLMState } from "./type";
import Dialog from "./Dialog";
import { DetectionCategory, getLanguageTexts, LanguageText } from "../settings";
import VlmSelector from "./VlmSelector";

function VlmImagePanel({
  imageSrc,
  alt,
}: {
  imageSrc: string | null,
  alt: string,
}) {
  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        maxWidth: "388px",
        aspectRatio: "16 / 10",
        borderRadius: "20px",
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.03)",
        backgroundColor: "#2A2A2A",
        boxShadow: "none",
      }}
    >
      {imageSrc != null &&
        <Box
          component="img"
          src={imageSrc}
          alt={alt}
          sx={{
            display: "block",
            width: "100%",
            height: "100%",
            objectFit: "contain",
          }}
        />
      }
    </Box>
  );
}

type VlmColumnProps = {
  title: string,
  client: LLMClient,
  language: string,
  texts: LanguageText,
  isAutoMode: boolean,
  isConnected: boolean,
  waitingText: string,
  detectionThreshold: number,
  setThreshold: (value: number) => void,
  vlmModels: string[],
  currentVlmModel: string,
  isVlmModelSwitching: boolean,
  isInteractionBlocked: boolean,
  changeVlmModel: (modelId: string) => void,
};

function VlmColumn({
  title,
  client,
  language,
  texts,
  isAutoMode,
  isConnected,
  waitingText,
  detectionThreshold,
  setThreshold,
  vlmModels,
  currentVlmModel,
  isVlmModelSwitching,
  isInteractionBlocked,
  changeVlmModel,
}: VlmColumnProps) {
  const scrollGridRef = useRef<HTMLDivElement | null>(null);
  const isWaitingForDetection = client.state == LLMState.IDLE;
  const isWaitingWithoutDialog = !isConnected || (isWaitingForDetection && client.dialog.length == 0);
  const isWaitingAfterDialog = isConnected && isWaitingForDetection && client.dialog.length > 0;
  const resolvedWaitingText = !isConnected
    ? texts.statusConnecting
    : isAutoMode
      ? waitingText
      : texts.statusWaitingForManualSelection;

  return (
    <Grid2
      container
      direction="column"
      alignItems="stretch"
      wrap="nowrap"
      size="grow"
      sx={{
        height: "100%",
        minHeight: 0,
        minWidth: 0,
      }}
    >
      <Typography
        sx={{
          color: "#FFFFFF",
          fontSize: "21px",
          fontWeight: 500,
          lineHeight: "120%",
          letterSpacing: "-0.2px",
          textAlign: "center",
          marginBottom: "18px",
        }}
      >
        {title}
      </Typography>
      <Grid2 container alignItems="center" columnSpacing="16px" wrap="nowrap" sx={{ width: "100%" }}>
        <Grid2
          container
          alignItems="center"
          columnSpacing="17px"
          sx={{
            flex: "1 1 auto",
            minWidth: 0,
            minHeight: "48px",
            padding: "8px 24px",
            borderRadius: "57px",
            backgroundColor: "#212631",
          }}
        >
          <Grid2
            container
            alignItems={"center"}
            size="grow"
            sx={{ minWidth: 0 }}
          >
            <Tooltip title={texts.detectionThresholdLabel} placement="top" arrow>
              <Slider
                value={detectionThreshold}
                disabled={isInteractionBlocked}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => value.toFixed(2)}
                onChange={(_, value) => setThreshold(value as number)}
                sx={{
                  padding: "0px !important",
                  color: "#2362DB",
                  "& .MuiSlider-thumb": {
                    width: 18,
                    height: 18,
                    backgroundColor: "#FFFFFF",
                  },
                  "& .MuiSlider-rail": { height: 6, backgroundColor: "rgba(255,255,255,0.25)" },
                  "& .MuiSlider-track": { height: 6 },
                }}
              />
            </Tooltip>
          </Grid2>
          <Typography
            sx={{
              color: "#FFFFFF",
              fontSize: "14px",
              fontWeight: 400,
              textAlign: "right",
            }}
          >
            {detectionThreshold.toFixed(2)}
          </Typography>
        </Grid2>
        <Grid2
          container
          alignItems="center"
          sx={{
            flex: "0 0 auto",
            padding: "7px 8px",
          }}
        >
          <VlmSelector
            models={vlmModels}
            currentModel={currentVlmModel}
            disabled={isInteractionBlocked || isVlmModelSwitching}
            texts={texts}
            changeModel={changeVlmModel}
          />
        </Grid2>
      </Grid2>
      <Grid2
        container
        size="grow"
        direction="column"
        wrap="nowrap"
        justifyContent={client.dialog.length == 0 ? "center" : undefined}
        alignItems="stretch"
        rowSpacing="22px"
        sx={{
          width: "100%",
          minHeight: 0,
          margin: "35px 0 0",
        }}
      >
        <Grid2
          container
          sx={{
            flex: "0 0 auto",
            minWidth: 0,
            alignItems: "center",
            justifyContent: "center",
            alignSelf: "stretch",
            zIndex: 1,
          }}
        >
          <VlmImagePanel
            imageSrc={client.image}
            alt={title}
          />
        </Grid2>
        <Grid2
          container
          size="grow"
          direction="column"
          wrap="nowrap"
          justifyContent={client.dialog.length == 0 ? "center" : undefined}
          alignItems="stretch"
          rowSpacing="34px"
          sx={{
            minWidth: 0,
            minHeight: 0,
            flex: "1 1 auto",
            overflowY: "auto",
            padding: "10px 8px 0",
          }}
          ref={scrollGridRef}
        >
        {isWaitingWithoutDialog &&
          <Grid2 container justifyContent="center" alignItems="center" sx={{ minHeight: "100%" }}>
            <Typography
              sx={{
                color: "#FFFFFF",
                fontSize: "19px",
                fontWeight: 400,
                lineHeight: "170%",
                textAlign: "center",
                letterSpacing: "-0.1px",
              }}
            >
              {resolvedWaitingText}
            </Typography>
          </Grid2>
        }
        <Dialog
          client={client}
          language={language}
          scrollGridRef={scrollGridRef}
        />
        </Grid2>
      </Grid2>
      {isWaitingAfterDialog &&
        <Grid2
          container
          justifyContent="center"
          alignItems="center"
          sx={{
            width: "100%",
            paddingTop: "8px",
          }}
        >
          {isAutoMode
            ? <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  padding: "10px 28px",
                  borderRadius: "57px",
                  backgroundColor: "rgba(74, 222, 128, 0.12)",
                  border: "1px solid rgba(74, 222, 128, 0.5)",
                }}
              >
                <Typography
                  sx={{
                    color: "#4ADE80",
                    fontSize: "18px",
                    fontWeight: 600,
                    lineHeight: "140%",
                    textAlign: "center",
                    letterSpacing: "-0.2px",
                  }}
                >
                  ✓ {texts.statusAnswerHold}
                </Typography>
              </Box>
            : <Typography
                sx={{
                  color: "#C8D1DC",
                  fontSize: "18px",
                  fontWeight: 500,
                  lineHeight: "140%",
                  textAlign: "center",
                  letterSpacing: "-0.2px",
                }}
              >
                {resolvedWaitingText}
              </Typography>
          }
        </Grid2>
      }
    </Grid2>
  );
}

export type ChatProps = {
  clients: Record<DetectionCategory, LLMClient>,
  language: string,
  isAutoMode: boolean,
  isConnected: Record<DetectionCategory, boolean>,
  thresholds: Record<DetectionCategory, number>,
  setThreshold: Record<DetectionCategory, (value: number) => void>,
  vlmModels: Record<DetectionCategory, string[]>,
  currentVlmModel: Record<DetectionCategory, string>,
  isVlmModelSwitching: Record<DetectionCategory, boolean>,
  isInteractionBlocked: boolean,
  changeVlmModel: (category: DetectionCategory, modelId: string) => void,
};

export default function Chat({
  clients,
  language,
  isAutoMode,
  isConnected,
  thresholds,
  setThreshold,
  vlmModels,
  currentVlmModel,
  isVlmModelSwitching,
  isInteractionBlocked,
  changeVlmModel,
}: ChatProps) {
  const texts = getLanguageTexts(language);
  const waitingText = texts.statusWaitingForDetection;
  const columnTitles: Record<DetectionCategory, string> = {
    weapon: texts.weaponPanelTitle,
    fall: texts.fallPanelTitle,
  };

  return (
    <Grid2
      container
      direction="row"
      wrap="nowrap"
      size="grow"
      columnSpacing="48px"
      sx={{
        width: "100%",
        height: "100%",
        padding: 0,
        minHeight: 0,
        minWidth: 0,
      }}
    >
      {(["weapon", "fall"] as DetectionCategory[]).map((category) => (
        <VlmColumn
          key={category}
          title={columnTitles[category]}
          client={clients[category]}
          language={language}
          texts={texts}
          isAutoMode={isAutoMode}
          isConnected={isConnected[category]}
          waitingText={waitingText}
          detectionThreshold={thresholds[category]}
          setThreshold={setThreshold[category]}
          vlmModels={vlmModels[category]}
          currentVlmModel={currentVlmModel[category]}
          isVlmModelSwitching={isVlmModelSwitching[category]}
          isInteractionBlocked={isInteractionBlocked}
          changeVlmModel={(modelId) => changeVlmModel(category, modelId)}
        />
      ))}
    </Grid2>
  )
}
