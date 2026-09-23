import { mockBlend, mockExtreme, mockLeadTimes, mockRegions, mockReplayDetails, mockReplaySummaries, mockSources, mockVariables, mockVerification, mockWeightMap } from "./mock";
import type { BlendResponse, ExtremeGuidanceResponse, HealthResponse, Region, ReplayEventDetail, ReplayEventSummary, SourceMeta, VariableMeta, VerificationSummary, WeightMapResponse, RegionCode, VariableName, Season, Regime } from "./types";

export const API_BASE_URL = import.meta.env["VITE_API_BASE_URL"] ?? "/api/v1";
export class ApiError extends Error { constructor(message: string, public status?: number) { super(message); } }
async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { signal: signal ?? null, headers: { Accept: "application/json" } });
  if (!response.ok) { let detail = `Request failed (${response.status})`; try { const body = await response.json(); detail = body.detail ?? detail; } catch { /* retain status */ } throw new ApiError(detail, response.status); }
  // The same-origin gateway serves demonstration fixtures when no live backend is reachable.
  if (response.headers.get("x-skillblend-data") === "fixture") throw new ApiError("Live backend unavailable — fixture data", 503);
  return response.json() as Promise<T>;
}
export const liveApi = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", signal), regions: (signal?: AbortSignal) => request<Region[]>("/regions", signal),
  variables: (signal?: AbortSignal) => request<VariableMeta[]>("/variables", signal), leadTimes: (signal?: AbortSignal) => request<number[]>("/lead-times", signal),
  sources: (signal?: AbortSignal) => request<Record<string, SourceMeta>>("/sources", signal),
  blend: (r: RegionCode, v: VariableName, l: number, signal?: AbortSignal) => request<BlendResponse>(`/forecast/blend?region=${r}&variable=${v}&lead_hours=${l}`, signal),
  weights: (r: RegionCode, v: VariableName, s: Season, g: Regime, signal?: AbortSignal) => request<WeightMapResponse>(`/weights/map?region=${r}&variable=${v}&season=${s}&regime=${g}`, signal),
  verification: async (signal?: AbortSignal) => (await request<VerificationSummary[]>("/skill/verification", signal)).filter((row) => row.relative_csi_improvement != null && row.best_single_model_csi != null && row.skillblend_csi != null),
  extremes: (r: RegionCode, l: number, signal?: AbortSignal) => request<ExtremeGuidanceResponse>(`/extreme/guidance?region=${r}&lead_hours=${l}`, signal),
  replayEvents: (signal?: AbortSignal) => request<ReplayEventSummary[]>("/replay/events", signal), replayEvent: (id: string, signal?: AbortSignal) => request<ReplayEventDetail>(`/replay/events/${encodeURIComponent(id)}`, signal),
};
export const fixtureApi = {
  regions: async () => mockRegions, variables: async () => mockVariables, leadTimes: async () => mockLeadTimes, sources: async () => mockSources,
  blend: async (r: RegionCode, v: VariableName, l: number) => mockBlend(r,v,l), weights: async (r: RegionCode,v: VariableName,s: Season,g: Regime) => mockWeightMap(r,v,s,g),
  verification: async () => mockVerification, extremes: async (r: RegionCode,l: number) => mockExtreme(r,l), replayEvents: async () => mockReplaySummaries,
  replayEvent: async (id: string): Promise<ReplayEventDetail> => {
    const match = mockReplayDetails.find((event) => event.event_id === id) ?? mockReplayDetails[0];
    if (!match) throw new ApiError("No replay fixtures available");
    return match;
  },
};
export async function withFallback<T>(live: () => Promise<T>, fixture: () => Promise<T>): Promise<{ data: T; mode: "live" | "mock" }> {
  try { return { data: await live(), mode: "live" }; } catch { return { data: await fixture(), mode: "mock" }; }
}
