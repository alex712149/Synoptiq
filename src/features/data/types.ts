export type RegionCode = "KWG" | "BOB" | "IGP";
export type VariableName = "precipitation" | "temperature" | "wind_speed";
export type ModelName = "GFS" | "IFS" | "AIFS";
export type Season = "winter" | "pre_monsoon" | "sw_monsoon" | "post_monsoon";
export type Regime = "active_monsoon" | "break_monsoon" | "western_disturbance" | "depression" | "normal";

export interface Region { code: RegionCode; name: string; lat: number; lon: number; emphasis: string; coastal: boolean }
export interface VariableMeta { variable: VariableName; unit: string; extreme_threshold: number }
export interface SourceMeta { kind: string; provider: string; role: string }
export interface SourceContribution { model: ModelName; forecast_value: number; weight: number | null; historical_skill: number | null }
export interface TrustBreakdown { historical_skill_component: number; disagreement_component: number; lead_time_component: number; regime_stability_component: number; data_quality_component: number; trust_score: number }
export interface ExplanationDriver { feature: string; contribution: number; direction: "increases_trust" | "decreases_trust" | "increases_weight" | "decreases_weight"; detail: string }
export interface BlendResponse {
  region: RegionCode; region_name: string; variable: VariableName; unit: string; run_time: string; valid_time: string; lead_hours: number;
  season: Season; regime: Regime; regime_probs: Record<Regime, number>; sources: SourceContribution[]; disagreement: number;
  raw_blend_value: number; bias_corrected_value: number; final_value: number; trust: TrustBreakdown; bust_probability: number;
  bust_flag: boolean; abstain: boolean; explanation: ExplanationDriver[]; fallback_used: boolean; provenance: Record<string, string>;
}
export interface WeightMapPoint { lead_hours: number; weights: Record<ModelName, number>; trust_score: number }
export interface WeightMapResponse { region: RegionCode; variable: VariableName; regime: Regime; season: Season; points: WeightMapPoint[] }
export interface VerificationSummary { region: RegionCode; variable: VariableName; threshold: number; best_single_model: ModelName; best_single_model_csi: number; skillblend_csi: number; relative_csi_improvement: number; meets_target: boolean; target_relative_csi_improvement: number }
export interface ExtremeProbability { variable: VariableName; threshold: number; unit: string; probability: number; calibrated: boolean }
export interface ExtremeGuidanceResponse { region: RegionCode; lead_hours: number; valid_time: string; guidance: ExtremeProbability[] }
export interface ReplayEventSummary { event_id: string; region: RegionCode; variable: VariableName; valid_time: string; label: string; headline: string }
export interface ReplayEventDetail extends ReplayEventSummary { unit: string; lead_hours: number; raw_sources: SourceContribution[]; naive_average: number; single_model_choice: { model: string; forecast_value: number }; skillblend_blend: number; reference_value: number; naive_average_error: number; single_model_error: number; skillblend_error: number; narrative: string[] }
export interface HealthResponse { status: "ok" | "not_seeded"; historical_data_cached: boolean; blend_model_trained: boolean; bust_model_trained: boolean; replay_events_cached: boolean }
export type DataMode = "live" | "mock";
