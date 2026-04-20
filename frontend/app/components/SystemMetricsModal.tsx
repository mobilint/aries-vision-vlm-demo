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
  available: boolean,
  temperature_c: number | null,
  utilization_pct: number | null,
  power_w: number | null,
  total_power_w?: number | null,
  source: string,
};

type SystemMetrics = {
  timestamp: number | null,
  cpu: DeviceMetrics | null,
  npu: DeviceMetrics | null,
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
}: {
  icon: ReactNode,
  label: string,
  value: string,
}) {
  return (
    <Box
      sx={{
        padding: "16px 18px",
        borderRadius: "18px",
        border: "1px solid rgba(128, 153, 184, 0.16)",
        background: "linear-gradient(180deg, rgba(22, 31, 43, 0.96) 0%, rgba(12, 18, 28, 0.98) 100%)",
      }}
    >
      <Grid2 container alignItems="center" columnSpacing="10px" sx={{ marginBottom: "12px" }}>
        <Grid2>{icon}</Grid2>
        <Grid2>
          <Typography sx={{ color: "#97A8BA", fontSize: "13px", fontWeight: 600 }}>
            {label}
          </Typography>
        </Grid2>
      </Grid2>
      <Typography
        sx={{
          color: "#FFFFFF",
          fontFamily: "CascadiaCode",
          fontSize: "28px",
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
  const height = 180;
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
          padding: "18px",
          borderRadius: "18px",
          border: "1px solid rgba(128, 153, 184, 0.16)",
          background: "linear-gradient(180deg, rgba(18, 27, 39, 0.96) 0%, rgba(10, 15, 23, 0.98) 100%)",
          minHeight: `${height + 40}px`,
        }}
      >
        <Typography sx={{ color: "#E8EEF6", fontSize: "16px", fontWeight: 600, marginBottom: "12px" }}>
          {title}
        </Typography>
        <Typography sx={{ color: "#8395A8", fontSize: "14px" }}>
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
        padding: "18px",
        borderRadius: "18px",
        border: "1px solid rgba(128, 153, 184, 0.16)",
        background: "linear-gradient(180deg, rgba(18, 27, 39, 0.96) 0%, rgba(10, 15, 23, 0.98) 100%)",
      }}
    >
      <Grid2 container justifyContent="space-between" alignItems="center" sx={{ marginBottom: "10px" }}>
        <Typography sx={{ color: "#E8EEF6", fontSize: "16px", fontWeight: 600 }}>
          {title}
        </Typography>
        <Typography sx={{ color: color, fontSize: "14px", fontWeight: 600, fontFamily: "CascadiaCode" }}>
          {formatNumber(series[series.length - 1]?.value, suffix)}
        </Typography>
      </Grid2>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "180px", display: "block" }}>
        <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={paddingTop + chartHeight} stroke="rgba(128, 153, 184, 0.26)" />
        <line x1={paddingLeft} y1={paddingTop + chartHeight} x2={paddingLeft + chartWidth} y2={paddingTop + chartHeight} stroke="rgba(128, 153, 184, 0.26)" />
        <line x1={paddingLeft} y1={paddingTop + chartHeight / 2} x2={paddingLeft + chartWidth} y2={paddingTop + chartHeight / 2} stroke="rgba(128, 153, 184, 0.12)" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke={color} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
        {points.map((point) => (
          <circle key={`${point.timestamp}-${point.value}`} cx={point.x} cy={point.y} r="3.5" fill={color} />
        ))}
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
      <Typography sx={{ color: "#6D7E90", fontSize: "12px", marginTop: "8px" }}>
        X: time, Y: value
      </Typography>
    </Box>
  );
}

function DeviceSection({
  title,
  metrics,
  history,
}: {
  title: string,
  metrics: DeviceMetrics | null,
  history: SystemMetrics[],
}) {
  const isAvailable = metrics?.available ?? false;
  const temperatureSeries = buildMetricSeries(history, (item) => title === "CPU"
    ? item.cpu?.temperature_c
    : item.npu?.temperature_c);
  const utilizationSeries = buildMetricSeries(history, (item) => title === "CPU"
    ? item.cpu?.utilization_pct
    : item.npu?.utilization_pct);
  const powerSeries = buildMetricSeries(history, (item) => title === "CPU"
    ? item.cpu?.power_w
    : item.npu?.power_w);

  return (
    <Box
      sx={{
        padding: "24px",
        borderRadius: "24px",
        border: "1px solid rgba(132, 160, 196, 0.18)",
        background: "linear-gradient(180deg, rgba(14, 22, 33, 0.98) 0%, rgba(8, 13, 21, 0.98) 100%)",
      }}
    >
      <Grid2 container justifyContent="space-between" alignItems="center" sx={{ marginBottom: "18px" }}>
        <Typography sx={{ color: "#FFFFFF", fontSize: "26px", fontWeight: 600 }}>
          {title}
        </Typography>
        <Chip
          label={isAvailable ? "Online" : "Unavailable"}
          sx={{
            color: isAvailable ? "#D9F7E9" : "#D6DDE6",
            backgroundColor: isAvailable ? "rgba(31, 145, 94, 0.18)" : "rgba(120, 136, 153, 0.16)",
            borderRadius: "999px",
            height: "30px",
            fontWeight: 600,
          }}
        />
      </Grid2>
      <Grid2 container spacing="14px" sx={{ marginBottom: "18px" }}>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <MetricTile
            icon={<ThermostatRoundedIcon sx={{ color: "#FFB167", fontSize: "20px" }} />}
            label="Temperature"
            value={formatNumber(metrics?.temperature_c, " C")}
          />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <MetricTile
            icon={<MemoryRoundedIcon sx={{ color: "#77B8FF", fontSize: "20px" }} />}
            label="Utilization"
            value={formatNumber(metrics?.utilization_pct, "%")}
          />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <MetricTile
            icon={<BoltRoundedIcon sx={{ color: "#FFD56A", fontSize: "20px" }} />}
            label="Power"
            value={formatNumber(metrics?.power_w, "W")}
          />
        </Grid2>
      </Grid2>
      <Grid2 container spacing="14px">
        <Grid2 size={{ xs: 12, md: 4 }}>
          <MetricChart title="Temperature" suffix=" C" color="#FFB167" series={temperatureSeries} />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <MetricChart title="Utilization" suffix="%" color="#77B8FF" series={utilizationSeries} />
        </Grid2>
        <Grid2 size={{ xs: 12, md: 4 }}>
          <MetricChart title="Power" suffix="W" color="#FFD56A" series={powerSeries} />
        </Grid2>
      </Grid2>
      {metrics?.total_power_w != null &&
        <Typography
          sx={{
            color: "#90A0B1",
            fontSize: "13px",
            marginTop: "14px",
            fontFamily: "CascadiaCode",
          }}
        >
          Total board power: {formatNumber(metrics.total_power_w, "W")}
        </Typography>
      }
      <Typography
        sx={{
          color: "#6D7E90",
          fontSize: "12px",
          marginTop: "8px",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
        }}
      >
        Source: {metrics?.source ?? "unknown"}
      </Typography>
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
            maxHeight: "calc(100vh - 48px)",
            overflowX: "hidden",
            overflowY: "auto",
          },
        },
      }}
    >
      <Box sx={{ padding: "28px 30px 30px" }}>
        <Grid2 container justifyContent="space-between" alignItems="center">
          <Box>
            <Typography sx={{ color: "#FFFFFF", fontSize: "34px", fontWeight: 600 }}>
              {texts.systemMetricsTitle}
            </Typography>
            <Typography sx={{ color: "#93A2B4", fontSize: "14px", marginTop: "8px" }}>
              {texts.systemMetricsDescription}
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
            marginTop: "16px",
            fontFamily: "CascadiaCode",
          }}
        >
          {texts.systemMetricsUpdatedAt}: {updatedAt} | samples: {history.length}/20
        </Typography>
        <Divider sx={{ borderColor: "rgba(119, 139, 162, 0.16)", marginY: "20px" }} />
        <Grid2 container direction="column" rowSpacing="18px">
          {hasNpuMetrics &&
            <Grid2>
              <DeviceSection title="NPU" metrics={metrics.npu} history={history} />
            </Grid2>
          }
          {hasCpuMetrics &&
            <Grid2>
              <DeviceSection title="CPU" metrics={metrics.cpu} history={history} />
            </Grid2>
          }
          {!hasCpuMetrics && !hasNpuMetrics &&
            <DeviceSection title="NPU" metrics={metrics.npu} history={history} />
          }
        </Grid2>
      </Box>
    </Dialog>
  );
}
