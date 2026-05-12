import {
  Box,
  Chip,
  Dialog,
  Divider,
  Grid2,
  IconButton,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import ThermostatRoundedIcon from "@mui/icons-material/ThermostatRounded";
import BoltRoundedIcon from "@mui/icons-material/BoltRounded";
import MemoryRoundedIcon from "@mui/icons-material/MemoryRounded";
import type { ReactNode } from "react";

type DeviceMetrics = {
  name?: string | null,
  available: boolean,
  temperature_c: number | null,
  utilization_pct: number | null,
  power_w: number | null,
  dram_power_w?: number | null,
  power_status?: string | null,
  power_error?: string | null,
  p99_power_w?: number | null,
  max_power_w?: number | null,
  total_power_w?: number | null,
  power_samples?: number | null,
  used_mb?: number | null,
  total_mb?: number | null,
  available_mb?: number | null,
  source: string,
};

type SystemMetrics = {
  timestamp: number | null,
  cpu: DeviceMetrics | null,
  npu: DeviceMetrics | null,
  ram?: DeviceMetrics | null,
};

type MetricPoint = {
  timestamp: number,
  value: number,
};

function formatNumber(value: number | null | undefined, suffix: string): string {
  if (value == null || Number.isNaN(value))
    return "N/A";

  return `${value.toFixed(1)}${suffix}`;
}

function formatMemory(valueMb: number | null | undefined): string {
  if (valueMb == null || Number.isNaN(valueMb))
    return "N/A";

  if (valueMb >= 1024)
    return `${(valueMb / 1024).toFixed(1)}GB`;

  return `${valueMb.toFixed(0)}MB`;
}

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function MetricTile({
  icon,
  label,
  value,
  hidden = false,
}: {
  icon: ReactNode,
  label: string,
  value: string,
  hidden?: boolean,
}) {
  return (
    <Box
      sx={{
        visibility: hidden ? "hidden" : "visible",
        padding: "clamp(7px, 0.8vh, 10px) 12px",
        borderRadius: "14px",
        border: "1px solid rgba(128, 153, 184, 0.16)",
        background: "linear-gradient(180deg, rgba(22, 31, 43, 0.96) 0%, rgba(12, 18, 28, 0.98) 100%)",
      }}
    >
      <Grid2 container alignItems="center" columnSpacing="8px" sx={{ marginBottom: "clamp(4px, 0.55vh, 8px)" }}>
        <Grid2>{icon}</Grid2>
        <Grid2>
          <Typography sx={{ color: "#97A8BA", fontSize: "11px", fontWeight: 600 }}>
            {label}
          </Typography>
        </Grid2>
      </Grid2>
      <Typography
        sx={{
          color: "#FFFFFF",
          fontFamily: "CascadiaCode",
          fontSize: "clamp(17px, 1.7vh, 20px)",
          fontWeight: 600,
          lineHeight: 1.1,
        }}
      >
        {value}
      </Typography>
    </Box>
  );
}

function buildMetricSeries(
  history: SystemMetrics[],
  selector: (metrics: SystemMetrics) => number | null | undefined,
): MetricPoint[] {
  return history.flatMap((item) => {
    if (item.timestamp == null) {
      return [];
    }

    const value = selector(item);
    if (value == null || Number.isNaN(value)) {
      return [];
    }

    return [{ timestamp: item.timestamp, value }];
  });
}

function MetricChart({
  title,
  suffix,
  color,
  series,
}: {
  title: string,
  suffix: string,
  color: string,
  series: MetricPoint[],
}) {
  const width = 340;
  const height = 96;
  const paddingLeft = 42;
  const paddingRight = 14;
  const paddingTop = 18;
  const paddingBottom = 30;
  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  if (series.length === 0) {
    return (
      <Box
        sx={{
          height: "100%",
          minHeight: 0,
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
          padding: "clamp(8px, 0.9vh, 12px)",
          borderRadius: "14px",
          border: "1px solid rgba(128, 153, 184, 0.16)",
          background: "linear-gradient(180deg, rgba(18, 27, 39, 0.96) 0%, rgba(10, 15, 23, 0.98) 100%)",
        }}
      >
        <Typography sx={{ color: "#E8EEF6", fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>
          {title}
        </Typography>
        <Typography sx={{ color: "#8395A8", fontSize: "12px", flex: 1 }}>
          No samples yet.
        </Typography>
      </Box>
    );
  }

  let minValue = Math.min(...series.map((point) => point.value));
  let maxValue = Math.max(...series.map((point) => point.value));
  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }

  const startTs = series[0].timestamp;
  const endTs = series[series.length - 1].timestamp;
  const timeSpan = Math.max(endTs - startTs, 1);
  const valueSpan = maxValue - minValue;

  const points = series.map((point) => {
    const x = paddingLeft + ((point.timestamp - startTs) / timeSpan) * chartWidth;
    const y = paddingTop + (1 - ((point.value - minValue) / valueSpan)) * chartHeight;
    return { x, y, ...point };
  });

  const path = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const midValue = minValue + valueSpan / 2;

  return (
    <Box
      sx={{
        height: "100%",
        minHeight: 0,
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        padding: "clamp(8px, 0.9vh, 12px)",
        borderRadius: "14px",
        border: "1px solid rgba(128, 153, 184, 0.16)",
        background: "linear-gradient(180deg, rgba(18, 27, 39, 0.96) 0%, rgba(10, 15, 23, 0.98) 100%)",
      }}
    >
      <Grid2 container justifyContent="space-between" alignItems="center" sx={{ marginBottom: "clamp(5px, 0.65vh, 10px)" }}>
        <Typography sx={{ color: "#E8EEF6", fontSize: "13px", fontWeight: 600 }}>
          {title}
        </Typography>
        <Typography sx={{ color: color, fontSize: "12px", fontWeight: 600, fontFamily: "CascadiaCode" }}>
          {formatNumber(series[series.length - 1]?.value, suffix)}
        </Typography>
      </Grid2>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", height: "100%", flex: "1 1 0", minHeight: 0, display: "block" }}
      >
        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={paddingTop + chartHeight} stroke="rgba(128, 153, 184, 0.26)" />
        <line x1={paddingLeft} y1={paddingTop + chartHeight} x2={paddingLeft + chartWidth} y2={paddingTop + chartHeight} stroke="rgba(128, 153, 184, 0.26)" />
        <line x1={paddingLeft} y1={paddingTop + chartHeight / 2} x2={paddingLeft + chartWidth} y2={paddingTop + chartHeight / 2} stroke="rgba(128, 153, 184, 0.12)" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke={color} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
        <text x={paddingLeft - 8} y={paddingTop + 4} textAnchor="end" fill="#8395A8" fontSize="11">
          {maxValue.toFixed(1)}
        </text>
        <text x={paddingLeft - 8} y={paddingTop + chartHeight / 2 + 4} textAnchor="end" fill="#8395A8" fontSize="11">
          {midValue.toFixed(1)}
        </text>
        <text x={paddingLeft - 8} y={paddingTop + chartHeight + 4} textAnchor="end" fill="#8395A8" fontSize="11">
          {minValue.toFixed(1)}
        </text>
        <text x={paddingLeft} y={height - 8} textAnchor="start" fill="#8395A8" fontSize="11">
          {formatTime(startTs)}
        </text>
        <text x={paddingLeft + chartWidth} y={height - 8} textAnchor="end" fill="#8395A8" fontSize="11">
          {formatTime(endTs)}
        </text>
      </svg>
      <Typography sx={{ color: "#6D7E90", fontSize: "10px", marginTop: "2px" }}>
        X: time, Y: value
      </Typography>
    </Box>
  );
}

function DeviceSection({
  title,
  metrics,
  history,
  type,
}: {
  title: string,
  metrics: DeviceMetrics | null,
  history: SystemMetrics[],
  type: "cpu" | "npu" | "ram",
}) {
  const isAvailable = metrics?.available ?? false;
  const temperatureSeries = buildMetricSeries(history, (item) => item[type]?.temperature_c);
  const utilizationSeries = buildMetricSeries(history, (item) => item[type]?.utilization_pct);
  const powerSeries = buildMetricSeries(history, (item) => item[type]?.power_w);
  const memoryUsedSeries = buildMetricSeries(history, (item) => item[type]?.used_mb);
  const isRam = type == "ram";
  const isRamPowerAvailable = isRam && metrics?.power_status !== "unavailable" && metrics?.power_w != null;

  return (
    <Box
      sx={{
        height: "100%",
        minHeight: 0,
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        padding: "clamp(10px, 1.15vh, 14px)",
        borderRadius: "18px",
        border: "1px solid rgba(132, 160, 196, 0.18)",
        background: "linear-gradient(180deg, rgba(14, 22, 33, 0.98) 0%, rgba(8, 13, 21, 0.98) 100%)",
      }}
    >
      <Grid2 container justifyContent="space-between" alignItems="flex-start" columnSpacing="12px" sx={{ marginBottom: "clamp(7px, 0.85vh, 10px)" }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography sx={{ color: "#FFFFFF", fontSize: "clamp(19px, 2vh, 22px)", fontWeight: 600 }}>
            {title}
          </Typography>
        </Box>
        <Box sx={{ minWidth: 0, maxWidth: "65%", display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "8px" }}>
          <Typography
            title={metrics?.name ?? undefined}
            sx={{
              color: "#9FB0C3",
              fontSize: "12px",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              textAlign: "right",
            }}
          >
            {metrics?.name ?? "Unknown device"}
          </Typography>
          <Chip
            label={isAvailable ? "Online" : "Unavailable"}
            sx={{
              flex: "0 0 auto",
              color: isAvailable ? "#D9F7E9" : "#D6DDE6",
              backgroundColor: isAvailable ? "rgba(31, 145, 94, 0.18)" : "rgba(120, 136, 153, 0.16)",
              borderRadius: "999px",
              height: "24px",
              fontSize: "11px",
              fontWeight: 600,
            }}
          />
        </Box>
      </Grid2>
      <Grid2 container spacing="8px" sx={{ marginBottom: "clamp(7px, 0.85vh, 10px)" }}>
        <Grid2 size={4}>
          <MetricTile
            icon={isRam
              ? <MemoryRoundedIcon sx={{ color: "#9FE870", fontSize: "20px" }} />
              : <ThermostatRoundedIcon sx={{ color: "#FFB167", fontSize: "20px" }} />
            }
            label={isRam ? "Used" : "Temperature"}
            value={isRam ? formatMemory(metrics?.used_mb) : formatNumber(metrics?.temperature_c, " C")}
          />
        </Grid2>
        <Grid2 size={4}>
          <MetricTile
            icon={<MemoryRoundedIcon sx={{ color: "#77B8FF", fontSize: "20px" }} />}
            label="Utilization"
            value={formatNumber(metrics?.utilization_pct, "%")}
          />
        </Grid2>
        <Grid2 size={4}>
          <MetricTile
            icon={isRam
              ? <BoltRoundedIcon sx={{ color: "#FFD56A", fontSize: "20px" }} />
              : <BoltRoundedIcon sx={{ color: "#FFD56A", fontSize: "20px" }} />
            }
            label={isRam ? "DRAM Power" : "Power"}
            value={formatNumber(metrics?.power_w, "W")}
            hidden={isRam && !isRamPowerAvailable}
          />
        </Grid2>
      </Grid2>
      <Box
        sx={{
          flex: "1 1 0",
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          gap: "clamp(6px, 0.75vh, 8px)",
        }}
      >
        {!isRam &&
          <Box sx={{ flex: "1 1 0", minHeight: 0 }}>
            <MetricChart title="Temperature" suffix=" C" color="#FFB167" series={temperatureSeries} />
          </Box>
        }
        <Box sx={{ flex: "1 1 0", minHeight: 0 }}>
          <MetricChart title="Utilization" suffix="%" color="#77B8FF" series={utilizationSeries} />
        </Box>
        <Box sx={{ flex: "1 1 0", minHeight: 0 }}>
          {isRam
            ? <MetricChart title="Used Memory" suffix="MB" color="#9FE870" series={memoryUsedSeries} />
            : <MetricChart title="Power" suffix="W" color="#FFD56A" series={powerSeries} />
          }
        </Box>
        {isRamPowerAvailable &&
          <Box sx={{ flex: "1 1 0", minHeight: 0 }}>
            <MetricChart title="DRAM Power" suffix="W" color="#FFD56A" series={powerSeries} />
          </Box>
        }
        {isRam && !isRamPowerAvailable &&
          <Box sx={{ flex: "1 1 0", minHeight: 0 }} />
        }
      </Box>
      <Box
        sx={{
          flex: "0 0 clamp(50px, 6vh, 66px)",
          minHeight: 0,
          overflow: "hidden",
          marginTop: "clamp(5px, 0.6vh, 8px)",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          gap: "clamp(2px, 0.28vh, 4px)",
        }}
      >
        {isRam && metrics?.total_mb != null &&
          <Typography
            sx={{
              color: "#90A0B1",
              fontSize: "11px",
              fontFamily: "CascadiaCode",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            Total memory: {formatMemory(metrics.total_mb)}
          </Typography>
        }
        {isRam && metrics?.available_mb != null &&
          <Typography
            sx={{
              color: "#90A0B1",
              fontSize: "11px",
              fontFamily: "CascadiaCode",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            Available memory: {formatMemory(metrics.available_mb)}
          </Typography>
        }
        {isRam && (metrics?.p99_power_w != null || metrics?.max_power_w != null) &&
          <Typography
            sx={{
              color: "#90A0B1",
              fontSize: "11px",
              fontFamily: "CascadiaCode",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            DRAM power p99/max: {formatNumber(metrics.p99_power_w, "W")} / {formatNumber(metrics.max_power_w, "W")}
          </Typography>
        }
        {metrics?.total_power_w != null &&
          <Typography
            sx={{
              color: "#90A0B1",
              fontSize: "11px",
              fontFamily: "CascadiaCode",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            Total board power: {formatNumber(metrics.total_power_w, "W")}
          </Typography>
        }
        <Typography
          sx={{
            color: "#6D7E90",
            fontSize: "10px",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          Source: {metrics?.source ?? "unknown"}
        </Typography>
      </Box>
    </Box>
  );
}

export default function SystemMetricsModal({
  open,
  onClose,
  texts,
  metrics,
  history,
}: {
  open: boolean,
  onClose: () => void,
  texts: {
    systemMetricsTitle: string,
    systemMetricsUpdatedAt: string,
    systemMetricsDescription: string,
  },
  metrics: SystemMetrics,
  history: SystemMetrics[],
}) {
  const updatedAt = metrics.timestamp == null
    ? "N/A"
    : formatTime(metrics.timestamp);
  const hasCpuMetrics = metrics.cpu != null;
  const hasNpuMetrics = metrics.npu != null;
  const ramMetrics = metrics.ram ?? null;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="xl"
      slotProps={{
        paper: {
          sx: {
            borderRadius: "30px",
            border: "1px solid rgba(124, 149, 180, 0.18)",
            background: "linear-gradient(180deg, rgba(10, 16, 25, 0.98) 0%, rgba(4, 8, 14, 0.99) 100%)",
            boxShadow: "0 28px 70px rgba(0, 0, 0, 0.45)",
            width: "calc(100vw - clamp(24px, 2.5vw, 48px))",
            maxWidth: "1600px",
            height: "calc(100vh - clamp(24px, 3.2vh, 48px))",
            maxHeight: "calc(100vh - clamp(24px, 3.2vh, 48px))",
            overflowX: "hidden",
            overflowY: "hidden",
          },
        },
      }}
    >
      <Box sx={{ padding: "clamp(14px, 1.8vh, 22px) 24px clamp(14px, 1.8vh, 24px)", height: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column" }}>
        <Grid2 container justifyContent="space-between" alignItems="center">
          <Box>
            <Typography sx={{ color: "#FFFFFF", fontSize: "clamp(24px, 2.8vh, 30px)", fontWeight: 600 }}>
              {texts.systemMetricsTitle}
            </Typography>
          </Box>
          <IconButton
            onClick={onClose}
            sx={{
              width: "42px",
              height: "42px",
              color: "#D7E1EC",
              backgroundColor: "rgba(148, 166, 190, 0.12)",
              border: "1px solid rgba(148, 166, 190, 0.18)",
            }}
          >
            <CloseIcon />
          </IconButton>
        </Grid2>
        <Typography
          sx={{
            color: "#B4C0CD",
            fontSize: "13px",
            marginTop: "clamp(8px, 1vh, 16px)",
            fontFamily: "CascadiaCode",
          }}
        >
          {texts.systemMetricsUpdatedAt}: {updatedAt} | samples: {history.length}/1440 | retention: 24h | refresh: 1m
        </Typography>
        <Divider sx={{ borderColor: "rgba(119, 139, 162, 0.16)", marginY: "clamp(8px, 1.1vh, 14px)" }} />
        <Grid2 container spacing="clamp(10px, 1.15vh, 14px)" sx={{ flex: "1 1 auto", minHeight: 0, overflow: "hidden" }}>
          <Grid2 size={4} sx={{ minHeight: 0 }}>
            <DeviceSection title="NPU" type="npu" metrics={hasNpuMetrics ? metrics.npu : null} history={history} />
          </Grid2>
          <Grid2 size={4} sx={{ minHeight: 0 }}>
            <DeviceSection title="CPU" type="cpu" metrics={hasCpuMetrics ? metrics.cpu : null} history={history} />
          </Grid2>
          <Grid2 size={4} sx={{ minHeight: 0 }}>
            <DeviceSection title="RAM" type="ram" metrics={ramMetrics} history={history} />
          </Grid2>
        </Grid2>
      </Box>
    </Dialog>
  );
}
