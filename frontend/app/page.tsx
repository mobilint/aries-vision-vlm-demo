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
import { defaultLLMClient, ImageDetectionOverlay, LLMClient, LLMState } from "./components/type";
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
  name?: string | null;
  available: boolean;
  temperature_c: number | null;
  utilization_pct: number | null;
  power_w: number | null;
  dram_power_w?: number | null;
  power_status?: string | null;
  power_error?: string | null;
  p99_power_w?: number | null;
  max_power_w?: number | null;
  total_power_w?: number | null;
  power_samples?: number | null;
  used_mb?: number | null;
  total_mb?: number | null;
  available_mb?: number | null;
  source: string;
};

type SystemMetrics = {
  timestamp: number | null;
  cpu: DeviceMetrics | null;
  npu: DeviceMetrics | null;
  ram?: DeviceMetrics | null;
};

type SystemMetricsHistory = SystemMetrics[];

type SystemMetricsPayload = SystemMetrics | {
  current: SystemMetrics;
  history: SystemMetrics[];
  sample_interval_seconds?: number;
  max_samples?: number;
};

const DEFAULT_SYSTEM_METRICS: SystemMetrics = {
  timestamp: null,
  cpu: null,
  npu: null,
  ram: null,
};

const SYSTEM_METRICS_POLL_MS = 60000;
const SYSTEM_METRICS_MAX_SAMPLES = 24 * 60;

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
  const [highlightedVisionChannelIndex, setHighlightedVisionChannelIndex] = useState<number | null>(null);
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
        setHighlightedVisionChannelIndex(bestDetection.channel.channel_index);

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
        ask(buildDetectionPrompt(bestDetection), selectedImage, {
          roi: bestDetection.detection.roi,
          imageWidth: bestDetection.channel.image_width,
          imageHeight: bestDetection.channel.image_height,
        });
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

  function isSystemMetricsEnvelope(payload: SystemMetricsPayload): payload is Exclude<SystemMetricsPayload, SystemMetrics> {
    return "current" in payload && "history" in payload;
  }

  function onSystemMetrics(payload: SystemMetricsPayload) {
    const currentMetrics = isSystemMetricsEnvelope(payload) ? payload.current : payload;
    const serverHistory = isSystemMetricsEnvelope(payload) ? payload.history : null;

    console.log("[system_metrics]", {
      payload,
      cpu_name: currentMetrics.cpu?.name,
      npu_name: currentMetrics.npu?.name,
      ram_name: currentMetrics.ram?.name,
      npu_available: currentMetrics.npu?.available,
      npu_temperature_c: currentMetrics.npu?.temperature_c,
      npu_utilization_pct: currentMetrics.npu?.utilization_pct,
      npu_power_w: currentMetrics.npu?.power_w,
      npu_total_power_w: currentMetrics.npu?.total_power_w,
      ram_utilization_pct: currentMetrics.ram?.utilization_pct,
      ram_power_w: currentMetrics.ram?.power_w,
      ram_dram_power_w: currentMetrics.ram?.dram_power_w,
      ram_power_samples: currentMetrics.ram?.power_samples,
      ram_used_mb: currentMetrics.ram?.used_mb,
      ram_total_mb: currentMetrics.ram?.total_mb,
    });
    setSystemMetrics(currentMetrics);

    if (serverHistory != null) {
      setSystemMetricsHistory(serverHistory.slice(-SYSTEM_METRICS_MAX_SAMPLES));
      return;
    }

    setSystemMetricsHistory((history) => {
      const nextHistory = [...history, currentMetrics];
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
    if (!isConnected || socket.current == null) {
      return;
    }

    socket.current.emit("system_metrics:get");
    const intervalId = window.setInterval(() => {
      socket.current?.emit("system_metrics:get");
    }, SYSTEM_METRICS_POLL_MS);

    return () => window.clearInterval(intervalId);
  }, [isConnected]);

  function ask(
    newQuestion: string,
    imageSrc?: string | null,
    imageDetectionOverlay?: ImageDetectionOverlay | null,
  ) {
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
        imageDetectionOverlay: imageDetectionOverlay ?? null,
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
    setHighlightedVisionChannelIndex(channelIndex);
    autoTriggerArmedRef.current = false;

    try {
      const payload = await loadVisionDetectionsSnapshot() ?? latestVisionDetectionsRef.current;
      const channel = payload?.channels.find((item) => item.channel_index == channelIndex);
      if (channel == null || channel.image_base64 == "") {
        console.warn("[vision manual] no latest frame for channel", channelIndex);
        return;
      }

      const bestDetection = channel.detections.reduce<VisionDetection | null>((best, detection) => {
        if (best == null || detection.confidence > best.confidence) {
          return detection;
        }

        return best;
      }, null);

      ask(
        buildManualChannelPrompt(channel),
        `data:image/jpeg;base64,${channel.image_base64}`,
        bestDetection == null ? null : {
          roi: bestDetection.roi,
          imageWidth: channel.image_width,
          imageHeight: channel.image_height,
        },
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
    setHighlightedVisionChannelIndex(null);

    setClient((client) => ({
      ...client,
      image: null,
      imageDetectionOverlay: null,
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
    setHighlightedVisionChannelIndex(null);
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
          backgroundColor: "#000000",
          padding: "62px 75px 68px 75px",
        }}
      >
        <Grid2
          container
          direction="column"
          wrap="nowrap"
          sx={{
            width: "100%",
            height: "100%",
            color: "#FFFFFF",
            overflow: "hidden",
          }}
        >
          <Grid2
            container
            justifyContent="space-between"
            alignItems="center"
            wrap="nowrap"
            padding="0"
          >
            <Grid2 container alignItems="center" wrap="nowrap" columnSpacing="26px" sx={{ minWidth: 0 }}>
              <Box
                component="img"
                src="/Mobilint_logo.png"
                alt="Mobilint"
                sx={{ width: "205px", height: "auto", display: "block" }}
              />
              <Typography sx={{ color: "#FFFFFF", fontSize: "36px", fontWeight: 400, lineHeight: 1 }}>
                ×
              </Typography>
              <Box
                component="img"
                src="/DFI_logo.png"
                alt="DFI"
                sx={{ width: "112px", height: "auto", display: "block" }}
              />
              <Typography
                sx={{
                  color: "#FFFFFF",
                  fontSize: "48px",
                  fontWeight: 500,
                  letterSpacing: "-1%",
                  lineHeight: "120%",
                  whiteSpace: "nowrap",
                }}
              >
                {texts.appTitle}
              </Typography>
            </Grid2>
            <Grid2
              container
              alignItems="center"
              columnSpacing="16px"
              sx={{ width: "fit-content" }}
            >
              <Button
                disableElevation
                disabled={!isPromptConfigReady || client.state != LLMState.IDLE}
                onClick={enableAutoMode}
                sx={{
                  padding: "15px 23px",
                  borderRadius: "999px",
                  fontWeight: 500,
                  fontSize: "14.71px",
                  border: "none",
                  color: "#FFFFFF",
                  backgroundColor: isAutoMode ? "#0B6BFF" : "#2A2A2A",
                  boxShadow: "none",
                  "&:hover": {
                    backgroundColor: isAutoMode ? "#0B6BFF" : "#343434",
                  },
                  "&.Mui-disabled": {
                    color: "#A0A0A0",
                    backgroundColor: isAutoMode ? "#164C9F" : "#242424",
                  },
                }}
              >
                {texts.autoLabel}
              </Button>
              <Button
                variant="outlined"
                onClick={() => setIsSystemMetricsOpen(true)}
                sx={{
                  padding: "15px 23px",
                  borderRadius: "999px",
                  color: "#FFFFFF",
                  backgroundColor: "#2A2A2A",
                  border: "none",
                  fontSize: "14.71px",
                  fontWeight: 500,
                  whiteSpace: "nowrap",
                  "&:hover": {
                    backgroundColor: "#343434",
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
          <Box sx={{ flex: "0 0 1px", width: "100%", backgroundColor: "rgba(255,255,255,0.36)", mt: "35px", mb: "69px" }} />
          <Grid2
            container
            direction="row"
            size="grow"
            columnSpacing="178px"
            wrap="nowrap"
            sx={{ minHeight: 0 }}
          >
            <Grid2
              container
              alignSelf="flex-start"
            >
              <VisionStream
                streamUrl={visionStreamUrl}
                layoutMetadata={visionLayout}
                highlightedChannelIndex={highlightedVisionChannelIndex}
                onChannelClick={handleManualChannelSelect}
                alt="8-channel vision stream"
              />
            </Grid2>
            <Grid2
              container
              size="grow"
            >
              <Chat
                client={client}
                language={language}
                isAutoMode={isAutoMode}
                detectionThreshold={detectionThreshold}
                setDetectionThreshold={setDetectionThreshold}
              />
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
