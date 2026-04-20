"use client";

import React, { useEffect, useRef, useState } from "react";
import io, { Socket } from "socket.io-client";
import {
  Box,
  Typography,
  Grid2,
  Button,
  createTheme,
  ThemeProvider,
} from "@mui/material";
import MonitorHeartRoundedIcon from "@mui/icons-material/MonitorHeartRounded";
import Chat from "./components/Chat";
import LanguageSwitcher from "./components/LanguageSwitcher";
import SystemMetricsModal from "./components/SystemMetricsModal";
import VisionStream from "./components/VisionStream";
import { defaultLLMClient, LLMClient, LLMState } from "./components/type";
import {
  AVAILABLE_LANGUAGES,
  DEFAULT_LANGUAGE,
  getLanguageTexts,
  loadPromptBundle,
} from "./settings";

const theme = createTheme({
  typography: {
    fontFamily: "Pretendard",
  },
});

const UI_WIDTH = 1920;
const UI_HEIGHT = 1080;
const VISION_DETECTIONS_POLL_MS = 500;
const VISION_LAYOUT_POLL_MS = 3000;
const DEFAULT_VISION_DETECTION_TRIGGER_THRESHOLD = 0.6;

type VisionDetection = {
  confidence: number;
  label: number;
  label_name: string;
  roi: [number, number, number, number];
};

type VisionDetectionChannel = {
  channel_index: number;
  feeder_index: number;
  model_index: number;
  has_detection: boolean;
  image_width: number;
  image_height: number;
  image_base64: string;
  detections: VisionDetection[];
};

type VisionDetectionResponse = {
  channels: VisionDetectionChannel[];
};

type BestVisionDetection = {
  channel: VisionDetectionChannel;
  detection: VisionDetection;
};

type VisionLayoutChannel = {
  channel_index: number;
  feeder_index: number;
  model_index: number;
  roi: [number, number, number, number];
};

type VisionLayoutResponse = {
  canvas: {
    width: number;
    height: number;
  };
  channel_count: number;
  channels: VisionLayoutChannel[];
  image_layout: {
    roi: [number, number, number, number];
  }[];
};

type DeviceMetrics = {
  available: boolean;
  temperature_c: number | null;
  utilization_pct: number | null;
  power_w: number | null;
  total_power_w?: number | null;
  source: string;
};

type SystemMetrics = {
  timestamp: number | null;
  cpu: DeviceMetrics | null;
  npu: DeviceMetrics | null;
};

type SystemMetricsHistory = SystemMetrics[];

const DEFAULT_SYSTEM_METRICS: SystemMetrics = {
  timestamp: null,
  cpu: null,
  npu: null,
};

const SYSTEM_METRICS_POLL_MS = 2000;
const SYSTEM_METRICS_MAX_SAMPLES = 20;

function formatDetectionList(detections: VisionDetection[]): string {
  if (detections.length == 0) {
    return "  - none";
  }

  return detections.map((item) => {
    return [
      "  - label: " + item.label,
      "    label_name: " + item.label_name,
      "    confidence: " + item.confidence.toFixed(4),
      "    roi: [" + item.roi.join(", ") + "]",
    ].join("\n");
  }).join("\n");
}

function buildDetectionPrompt(bestDetection: BestVisionDetection): string {
  const { channel, detection } = bestDetection;

  return [
    "detection_event:",
    `channel_index: ${channel.channel_index}`,
    `feeder_index: ${channel.feeder_index}`,
    `model_index: ${channel.model_index}`,
    `image_width: ${channel.image_width}`,
    `image_height: ${channel.image_height}`,
    "trigger_detection:",
    `  label: ${detection.label}`,
    `  label_name: ${detection.label_name}`,
    `  confidence: ${detection.confidence.toFixed(4)}`,
    `  roi: [${detection.roi.join(", ")}]`,
    "all_detections:",
    formatDetectionList(channel.detections),
  ].join("\n");
}

function buildManualChannelPrompt(channel: VisionDetectionChannel): string {
  return [
    "manual_channel_selection:",
    `channel_index: ${channel.channel_index}`,
    `feeder_index: ${channel.feeder_index}`,
    `model_index: ${channel.model_index}`,
    `image_width: ${channel.image_width}`,
    `image_height: ${channel.image_height}`,
    `has_detection: ${channel.has_detection}`,
    "latest_detections:",
    formatDetectionList(channel.detections),
  ].join("\n");
}

export default function Page() {
  const socket = useRef<Socket | null>(null);
  const languageRef = useRef(DEFAULT_LANGUAGE);
  const promptSyncRequestIdRef = useRef(0);
  const autoTriggerArmedRef = useRef(true);
  const latestVisionDetectionsRef = useRef<VisionDetectionResponse | null>(null);
  const pendingAskRef = useRef<{ question: string; imageSrc: string | null } | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [client, setClient] = useState<LLMClient>(defaultLLMClient);
  const [language, setLanguage] = useState(DEFAULT_LANGUAGE);
  const [isAutoMode, setIsAutoMode] = useState(true);
  const [isPromptConfigReady, setIsPromptConfigReady] = useState(false);
  const [isPromptConfigSyncing, setIsPromptConfigSyncing] = useState(false);
  const [visionStreamUrl, setVisionStreamUrl] = useState<string | null>(null);
  const [visionDetectionsUrl, setVisionDetectionsUrl] = useState<string | null>(null);
  const [visionLayoutUrl, setVisionLayoutUrl] = useState<string | null>(null);
  const [visionLayout, setVisionLayout] = useState<VisionLayoutResponse | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>(DEFAULT_SYSTEM_METRICS);
  const [systemMetricsHistory, setSystemMetricsHistory] = useState<SystemMetricsHistory>([]);
  const [isSystemMetricsOpen, setIsSystemMetricsOpen] = useState(false);
  const [detectionThreshold, setDetectionThreshold] = useState(
    DEFAULT_VISION_DETECTION_TRIGGER_THRESHOLD,
  );

  const texts = getLanguageTexts(language);

  useEffect(() => {
    languageRef.current = language;
    setClient((client) => ({ ...client, language }));
  }, [language]);

  useEffect(() => {
    const baseUrl = `${window.location.protocol}//${window.location.hostname}:8081`;
    setVisionStreamUrl(`${baseUrl}/stream.mjpg`);
    setVisionDetectionsUrl(`${baseUrl}/detections`);
    setVisionLayoutUrl(`${baseUrl}/layout`);
  }, []);

  useEffect(() => {
    if (visionLayoutUrl == null) {
      return;
    }

    let isCancelled = false;

    async function pollVisionLayout() {
      try {
        const layoutUrl = visionLayoutUrl;
        if (layoutUrl == null) {
          return;
        }

        const response = await fetch(layoutUrl, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Vision layout fetch failed: ${response.status}`);
        }

        const payload: VisionLayoutResponse = await response.json();
        if (!isCancelled) {
          setVisionLayout(payload);
        }
      } catch (error) {
        if (!isCancelled) {
          console.error("[vision layout]", error);
        }
      }
    }

    pollVisionLayout();
    const intervalId = window.setInterval(pollVisionLayout, VISION_LAYOUT_POLL_MS);

    return () => {
      isCancelled = true;
      window.clearInterval(intervalId);
    };
  }, [visionLayoutUrl]);

  async function loadVisionDetectionsSnapshot(): Promise<VisionDetectionResponse | null> {
    const detectionsUrl = visionDetectionsUrl;
    if (detectionsUrl == null) {
      return null;
    }

    const response = await fetch(detectionsUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Vision detections fetch failed: ${response.status}`);
    }

    const payload: VisionDetectionResponse = await response.json();
    latestVisionDetectionsRef.current = payload;
    return payload;
  }

  useEffect(() => {
    if (visionDetectionsUrl == null || !isConnected || !isPromptConfigReady) {
      return;
    }

    let isCancelled = false;

    function getBestVisionDetection(payload: VisionDetectionResponse): BestVisionDetection | null {
      let best: BestVisionDetection | null = null;

      for (const channel of payload.channels) {
        for (const detection of channel.detections) {
          if (detection.confidence <= detectionThreshold) {
            continue;
          }

          if (best == null || detection.confidence > best.detection.confidence) {
            best = { channel, detection };
          }
        }
      }

      return best;
    }

    async function pollVisionDetections() {
      try {
        const payload = await loadVisionDetectionsSnapshot();
        if (isCancelled || payload == null) {
          return;
        }

        const bestDetection = getBestVisionDetection(payload);
        if (bestDetection == null) {
          autoTriggerArmedRef.current = true;
          return;
        }

        if (!isAutoMode || client.state != LLMState.IDLE) {
          return;
        }

        if (!autoTriggerArmedRef.current) {
          return;
        }

        autoTriggerArmedRef.current = false;

        const selectedImage = `data:image/jpeg;base64,${bestDetection.channel.image_base64}`;
        console.log("[vision detections] selected for VLM", {
          channel_index: bestDetection.channel.channel_index,
          feeder_index: bestDetection.channel.feeder_index,
          model_index: bestDetection.channel.model_index,
          confidence: bestDetection.detection.confidence,
          label: bestDetection.detection.label,
          label_name: bestDetection.detection.label_name,
          roi: bestDetection.detection.roi,
        });
        ask(buildDetectionPrompt(bestDetection), selectedImage);
      } catch (error) {
        if (!isCancelled) {
          console.error("[vision detections]", error);
        }
      }
    }

    pollVisionDetections();
    const intervalId = window.setInterval(pollVisionDetections, VISION_DETECTIONS_POLL_MS);

    return () => {
      isCancelled = true;
      window.clearInterval(intervalId);
    };
  }, [visionDetectionsUrl, isConnected, isPromptConfigReady, client.state, detectionThreshold, isAutoMode]);

  function onConnect() {
    setIsConnected(true);
    setIsPromptConfigReady(false);
    setIsPromptConfigSyncing(true);
  }

  function onDisconnect() {
    setIsConnected(false);
    setIsPromptConfigReady(false);
    setIsPromptConfigSyncing(false);
  }

  function onModel(model: string) {
    setClient({ ...defaultLLMClient, model_id: model, language: languageRef.current });
  }

  function onTasks(tasks: number) {
    setClient((client) => ({ ...client, tasksNum: tasks }));
  }

  function onStart() {
    setClient((client) => {
      if (client.state == LLMState.ABORTING) {
        return client;
      }

      return { ...client, state: LLMState.ANSWERING };
    });
  }

  function onToken(token: string) {
    setClient((client) => {
      if (client.state != LLMState.ANSWERING || client.dialog.length == 0) {
        return client;
      }

      return {
        ...client,
        recentAnswer: client.recentAnswer == null ? token : client.recentAnswer + token,
      };
    });
  }

  function onEnd(isAborted: boolean) {
    setClient((client) => {
      if (client.state == LLMState.IDLE) {
        return client;
      }

      if (isAborted && pendingAskRef.current != null && client.state == LLMState.ASKING) {
        return client;
      }

      if (client.dialog.length == 0) {
        return {
          ...client,
          state: LLMState.IDLE,
          recentAnswer: null,
          tasksNum: 0,
        };
      }

      const newDialog = [...client.dialog];
      const recentAnswer = client.recentAnswer ?? "";
      newDialog[newDialog.length - 1].answer = recentAnswer + (isAborted ? " [ABORTED]" : "");

      return {
        ...client,
        dialog: newDialog,
        state: LLMState.IDLE,
        recentAnswer: null,
      };
    });
  }

  function onResetDone() {
    const pendingAsk = pendingAskRef.current;
    if (socket.current == null || pendingAsk == null) {
      return;
    }

    pendingAskRef.current = null;
    socket.current.emit("ask", pendingAsk.question, pendingAsk.imageSrc);
  }

  function onPromptConfigState(payload: { is_ready: boolean }) {
    setIsPromptConfigReady(payload.is_ready);
    if (payload.is_ready) {
      setIsPromptConfigSyncing(false);
    }
  }

  function onPromptConfigSaved() {
    setIsPromptConfigReady(true);
    setIsPromptConfigSyncing(false);
  }

  function onSystemMetrics(payload: SystemMetrics) {
    console.log("[system_metrics]", {
      payload,
      npu_available: payload.npu?.available,
      npu_temperature_c: payload.npu?.temperature_c,
      npu_utilization_pct: payload.npu?.utilization_pct,
      npu_power_w: payload.npu?.power_w,
      npu_total_power_w: payload.npu?.total_power_w,
    });
    setSystemMetrics(payload);
    setSystemMetricsHistory((history) => {
      const nextHistory = [...history, payload];
      return nextHistory.slice(-SYSTEM_METRICS_MAX_SAMPLES);
    });
  }

  useEffect(() => {
    socket.current = io(`${window.location.protocol == "https:" ? "wss" : "ws"}://${window.location.hostname}:5000`);
    socket.current.on("connect", onConnect);
    socket.current.on("disconnect", onDisconnect);
    socket.current.on("model", onModel);
    socket.current.on("tasks", onTasks);
    socket.current.on("start", onStart);
    socket.current.on("token", onToken);
    socket.current.on("end", onEnd);
    socket.current.on("reset_done", onResetDone);
    socket.current.on("prompt_config_state", onPromptConfigState);
    socket.current.on("prompt_config_saved", onPromptConfigSaved);
    socket.current.on("system_metrics", onSystemMetrics);

    return () => {
      if (socket.current) {
        socket.current.disconnect();
        socket.current.off("connect", onConnect);
        socket.current.off("disconnect", onDisconnect);
        socket.current.off("model", onModel);
        socket.current.off("tasks", onTasks);
        socket.current.off("start", onStart);
        socket.current.off("token", onToken);
        socket.current.off("end", onEnd);
        socket.current.off("reset_done", onResetDone);
        socket.current.off("prompt_config_state", onPromptConfigState);
        socket.current.off("prompt_config_saved", onPromptConfigSaved);
        socket.current.off("system_metrics", onSystemMetrics);
      }
    };
  }, []);

  useEffect(() => {
    if (!isConnected || socket.current == null) {
      return;
    }

    async function syncPromptBundle() {
      const requestId = ++promptSyncRequestIdRef.current;
      setIsPromptConfigSyncing(true);
      setIsPromptConfigReady(false);
      try {
        const promptBundle = await loadPromptBundle(language);
        if (promptSyncRequestIdRef.current != requestId) {
          return;
        }
        socket.current?.emit("prompt_config", promptBundle);
      } catch (error) {
        if (promptSyncRequestIdRef.current == requestId) {
          console.error("[prompt bundle]", error);
          setIsPromptConfigSyncing(false);
        }
      }
    }

    syncPromptBundle();
  }, [isConnected, language]);

  useEffect(() => {
    if (!isConnected || socket.current == null || !isSystemMetricsOpen) {
      return;
    }

    socket.current.emit("system_metrics:get");
    const intervalId = window.setInterval(() => {
      socket.current?.emit("system_metrics:get");
    }, SYSTEM_METRICS_POLL_MS);

    return () => window.clearInterval(intervalId);
  }, [isConnected, isSystemMetricsOpen]);

  function ask(newQuestion: string, imageSrc?: string | null) {
    if (socket.current == null || newQuestion == "" || !isPromptConfigReady) {
      return;
    }

    pendingAskRef.current = {
      question: newQuestion,
      imageSrc: imageSrc ?? null,
    };

    socket.current.emit("reset");

    setClient((client) => {
      return {
        ...client,
        dialog: [{ question: newQuestion, answer: null }],
        image: imageSrc ?? client.image,
        state: LLMState.ASKING,
        tasksNum: 0,
        recentAnswer: null,
      };
    });
  }

  async function handleManualChannelSelect(channelIndex: number) {
    if (socket.current == null || !isPromptConfigReady) {
      return;
    }

    setIsAutoMode(false);
    autoTriggerArmedRef.current = false;

    try {
      const payload = await loadVisionDetectionsSnapshot() ?? latestVisionDetectionsRef.current;
      const channel = payload?.channels.find((item) => item.channel_index == channelIndex);
      if (channel == null || channel.image_base64 == "") {
        console.warn("[vision manual] no latest frame for channel", channelIndex);
        return;
      }

      ask(
        buildManualChannelPrompt(channel),
        `data:image/jpeg;base64,${channel.image_base64}`,
      );
    } catch (error) {
      console.error("[vision manual]", error);
    }
  }

  function reset() {
    if (socket.current == null) {
      return;
    }

    autoTriggerArmedRef.current = true;
    pendingAskRef.current = null;

    setClient((client) => ({
      ...client,
      image: null,
      dialog: [],
      recentAnswer: null,
      tasksNum: 0,
      state: LLMState.IDLE,
    }));

    socket.current.emit("reset");
  }

  function changeLanguage(nextLanguage: string) {
    if (nextLanguage == language) {
      return;
    }

    setIsPromptConfigSyncing(true);
    reset();
    setLanguage(nextLanguage);
  }

  function enableAutoMode() {
    if (!isPromptConfigReady || client.state != LLMState.IDLE) {
      return;
    }

    autoTriggerArmedRef.current = true;
    setIsAutoMode(true);
  }

  if (isConnected == false) {
    return (
      <ThemeProvider theme={theme}>
        <Grid2
          container
          justifyContent="center"
          alignItems="center"
          sx={{ width: "100vw", height: "100vh", backgroundColor: "#111111" }}
        >
          <Typography sx={{ color: "#FFFFFF", fontSize: "24px", fontWeight: 500 }}>
            {texts.statusConnecting}
          </Typography>
        </Grid2>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          width: "100vw",
          height: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "auto",
          backgroundColor: "#05080C",
          p: 2,
        }}
      >
        <Grid2
          container
          direction="column"
          wrap="nowrap"
          rowSpacing={"24px"}
          sx={{
            width: UI_WIDTH,
            height: UI_HEIGHT,
            maxWidth: "100%",
            maxHeight: "100%",
            padding: "40px 40px 24px",
            background:
              "linear-gradient(180deg, rgba(10,16,24,0.98) 0%, rgba(5,8,12,1) 100%)",
            borderRadius: "28px",
            border: "1px solid rgba(87, 111, 142, 0.24)",
            boxShadow: "0 24px 60px rgba(0, 0, 0, 0.35)",
          }}
        >
          <Grid2
            container
            direction="row"
            size={"grow"}
            columnSpacing={"40px"}
            wrap="nowrap"
            sx={{ minHeight: 0 }}
          >
            <Grid2
              container
              sx={{
                flex: "0 0 auto",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
              }}
            >
              <VisionStream
                streamUrl={visionStreamUrl}
                layoutMetadata={visionLayout}
                onChannelClick={handleManualChannelSelect}
                alt="8-channel vision stream"
              />
            </Grid2>
            <Grid2
              container
              sx={{ flex: "1 1 auto", minWidth: 0, minHeight: 0 }}
            >
              <Grid2
                container
                size="grow"
                direction="column"
                wrap="nowrap"
              >
                <Grid2
                  container
                  justifyContent="space-between"
                  alignItems={"center"}
                >
                  <Typography
                    sx={{
                      fontWeight: 600,
                      fontSize: "48px",
                      lineHeight: "130%",
                      letterSpacing: "-0.2px",
                      verticalAlign: "middle",
                      color: "#FFFFFF",
                    }}
                  >
                    {texts.appTitle}
                  </Typography>
                  <Grid2 container alignItems="center" columnSpacing="12px" sx={{ width: "fit-content" }}>
                    <Button
                      disableElevation
                      disabled={!isPromptConfigReady || client.state != LLMState.IDLE}
                      onClick={enableAutoMode}
                      sx={{
                        minWidth: "auto",
                        height: "46px",
                        padding: "0 18px",
                        borderRadius: "999px",
                        textTransform: "none",
                        fontWeight: 700,
                        fontSize: "15px",
                        color: isAutoMode ? "#FFFFFF" : "#0B4EA2",
                        backgroundColor: isAutoMode ? "#0B4EA2" : "#FFFFFF",
                        border: isAutoMode ? "1px solid transparent" : "1px solid #D7DFEF",
                        boxShadow: "0 10px 30px rgba(13, 35, 67, 0.08)",
                        "&:hover": {
                          backgroundColor: isAutoMode ? "#0B4EA2" : "#F4F8FD",
                        },
                        "&.Mui-disabled": {
                          color: isAutoMode ? "#FFFFFF" : "#8EA1B8",
                          backgroundColor: isAutoMode ? "#7C96B8" : "#F5F7FA",
                          borderColor: "#E2E8F0",
                        },
                      }}
                    >
                      {texts.autoLabel}
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<MonitorHeartRoundedIcon sx={{ fontSize: "20px" }} />}
                      onClick={() => setIsSystemMetricsOpen(true)}
                      sx={{
                        height: "46px",
                        padding: "0 16px",
                        borderRadius: "14px",
                        color: "#F3F7FB",
                        borderColor: "rgba(109, 138, 173, 0.34)",
                        backgroundColor: "rgba(18, 28, 41, 0.82)",
                        fontSize: "14px",
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        "&:hover": {
                          borderColor: "rgba(127, 161, 203, 0.48)",
                          backgroundColor: "rgba(27, 39, 55, 0.96)",
                        },
                      }}
                    >
                      {texts.systemMetricsButton}
                    </Button>
                    <LanguageSwitcher
                      languages={[...AVAILABLE_LANGUAGES]}
                      currentLanguage={language}
                      disabled={client.state != LLMState.IDLE || isPromptConfigSyncing}
                      changeLanguage={changeLanguage}
                    />
                  </Grid2>
                </Grid2>
                <Chat
                  client={client}
                  language={language}
                  isAutoMode={isAutoMode}
                  detectionThreshold={detectionThreshold}
                  setDetectionThreshold={setDetectionThreshold}
                />
              </Grid2>
            </Grid2>
          </Grid2>
          {!isPromptConfigReady &&
            <Typography
              sx={{
                color: "#A8A8A8",
                fontSize: "14px",
                lineHeight: "130%",
              }}
            >
              {texts.statusPreparingPromptBundle}
            </Typography>
          }
        </Grid2>
        <SystemMetricsModal
          open={isSystemMetricsOpen}
          onClose={() => setIsSystemMetricsOpen(false)}
          texts={{
            systemMetricsTitle: texts.systemMetricsTitle,
            systemMetricsUpdatedAt: texts.systemMetricsUpdatedAt,
            systemMetricsDescription: texts.systemMetricsDescription,
          }}
          metrics={systemMetrics}
          history={systemMetricsHistory}
        />
      </Box>
    </ThemeProvider>
  );
}
