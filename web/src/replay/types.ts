export type Phase = "before" | "commitment" | "after";
export type Facing = "north" | "south" | "east" | "west";

export interface KitchenItem {
  kind: string;
  ingredients: string[];
  cooking_tick: number | null;
  ready: boolean;
}

export interface KitchenPlayer {
  physical_index: 0 | 1;
  role: "ego" | "partner";
  x: number;
  y: number;
  facing: Facing;
  held: KitchenItem | null;
}

export interface KitchenObject {
  x: number;
  y: number;
  item: KitchenItem;
}

export interface KitchenState {
  timestep: number;
  players: [KitchenPlayer, KitchenPlayer];
  objects: KitchenObject[];
}

export interface ReplayFrame {
  frame: number;
  action_step: number | null;
  state: KitchenState;
  ego_action: number | null;
  partner_action: number | null;
  sparse_reward: number;
  event_kinds: string[];
  phase: Phase;
}

export interface DiagnosticPoint {
  key: string;
  prefix: number | "pre_commitment" | "eventual";
  frame: number;
  phase: Phase;
  probability_left: number;
  probability_right: number;
  recommended_response_id: string;
  recommended_response_alias: string;
  estimated_expected_risk: number;
  selected_response_loss_for_true_mode: number;
  aggregate_event_dri: number | null;
}

export interface ReplayEvent {
  frame: number;
  step: number;
  kind: string;
  label: string;
  phase: Phase;
}

export interface ReplayManifest {
  schema_version: "beforecommit_replay_1.0.0";
  replay_id: string;
  title: string;
  short_title: string;
  category: string;
  caption: string;
  selection_note: string;
  representative: boolean;
  featured: boolean;
  source: {
    layout_id: "random3_m" | "small_corridor";
    partner_id: string;
    evidence_policy: string;
    environment_key: number;
    ego_seat: 0 | 1;
    policy_training_performed: false;
    checkpoint_hashes: Record<string, string>;
    source_artifact_hashes: Record<string, string>;
  };
  commitment: {
    definition: string;
    step: number | null;
    frame: number | null;
    reached: boolean;
  };
  terrain: string[][];
  partner_modes: { left: string; right: string };
  responses: Array<{ response_id: string; alias: string; optimal_for: string[] }>;
  summary: {
    sparse_return: number;
    first_delivery_step: number | null;
    steps_elapsed: number;
    prior_risk: number;
    aggregate_dri_primary: number | null;
    aggregate_dri_event: number | null;
    aggregate_residual_risk_primary: number;
    selected_response_loss_at_commitment: number;
    eventual_dri_primary: number | null;
  };
  frame_count: number;
  diagnostic_count: number;
  event_count: number;
  artifacts: Record<string, string>;
}

export interface ReplayBundle {
  manifest: ReplayManifest;
  frames: ReplayFrame[];
  diagnostics: DiagnosticPoint[];
  events: ReplayEvent[];
}

export interface CatalogItem {
  replay_id: string;
  title: string;
  short_title: string;
  category: string;
  caption: string;
  layout_id: string;
  evidence_policy: string;
  featured: boolean;
  representative: boolean;
  sparse_return: number;
  commitment_step: number | null;
  aggregate_dri_primary: number | null;
  eventual_dri_primary: number | null;
  frame_count: number;
}

export interface Catalog {
  collection_id: string;
  items: CatalogItem[];
  total: number;
}

export interface BenchmarkSummary {
  partner_count: number;
  method_family_count: number;
  layout_count: number;
  seats_per_layout: number;
  confirmation_episode_count: number;
  held_out_delta_r2: number;
  adjusted_dri_coefficient: number;
  coefficient_interval: [number, number];
  verdict: string;
  intervention_result: string;
}
