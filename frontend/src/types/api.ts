export interface EvidenceItem {
  chunk_id: string;
  claim: string;
}

export interface QueryResponse {
  answer: string;
  evidence: EvidenceItem[];
  confidence: number;
  recommended_action: string;
  human_review: boolean;
  intent: string;
  latency_ms: number;
}
