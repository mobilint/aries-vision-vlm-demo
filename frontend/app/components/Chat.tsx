import { Box, Grid2, Slider, Typography } from "@mui/material";
import { useRef } from "react";
import { LLMClient, LLMState } from "./type";
import Dialog from "./Dialog";
import { getLanguageTexts } from "../settings";

function VlmImagePanel({
  imageSrc,
  overlay,
  alt,
}: {
  imageSrc: string | null,
  overlay: {
    roi: [number, number, number, number],
    imageWidth: number,
    imageHeight: number,
  } | null,
  alt: string,
}) {
  const [x, y, width, height] = overlay?.roi ?? [0, 0, 0, 0];
  const cx = x + width / 2;
  const cy = y + height / 2;

  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        maxWidth: "388px",
        aspectRatio: overlay != null && overlay.imageWidth > 0 && overlay.imageHeight > 0
          ? `${overlay.imageWidth} / ${overlay.imageHeight}`
          : "16 / 10",
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
      {imageSrc != null && overlay != null && overlay.imageWidth > 0 && overlay.imageHeight > 0 &&
        <Box
          component="svg"
          viewBox={`0 0 ${overlay.imageWidth} ${overlay.imageHeight}`}
          preserveAspectRatio="xMidYMid meet"
          sx={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
        >
          <ellipse
            cx={cx}
            cy={cy}
            rx={Math.max(width * 0.58, 24)}
            ry={Math.max(height * 0.58, 24)}
            fill="rgba(255, 59, 48, 0.08)"
            stroke="rgba(255, 59, 48, 0.96)"
            strokeWidth="6"
          />
        </Box>
      }
    </Box>
  );
}

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
      alignItems="stretch"
      size="grow"
      sx={{
        width: "100%",
        height: "100%",
        padding: 0,
        minHeight: 0,
      }}
    >
      <Grid2
        container
        alignItems="center"
        columnSpacing="17px"
        sx={{
          width: "100%",
          minHeight: "68px",
          padding: "23px 44px",
          borderRadius: "57px",
          backgroundColor: "#212631",
        }}
      >
        <Typography
          sx={{
            color: "#FFFFFF",
            fontSize: "18px",
            fontWeight: 400,
            whiteSpace: "nowrap",
          }}
        >
          {texts.detectionThresholdLabel}
        </Typography>
        <Grid2
          container
          alignItems={"center"}
          size="grow"
          sx={{ minWidth: 0 }}
        >
          <Slider
            value={detectionThreshold}
            min={0}
            max={1}
            step={0.01}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => value.toFixed(2)}
            onChange={(_, value) => setDetectionThreshold(value as number)}
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
        </Grid2>
        <Typography
          sx={{
            color: "#FFFFFF",
              fontSize: "18px",
            fontWeight: 400,
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
        rowSpacing="22px"
        sx={{
          width: "100%",
          minHeight: 0,
          margin: "55px 0 0",
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
            overlay={client.imageDetectionOverlay}
            alt={texts.imagePanelTitle}
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
                fontSize: "24px",
                fontWeight: 400,
                lineHeight: "170%",
                textAlign: "center",
                letterSpacing: "-0.1px",
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
