-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Memories table
CREATE TABLE memories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL,
  memory_type VARCHAR(32) NOT NULL, -- 'episodic', 'semantic', 'procedural'
  content TEXT NOT NULL,
  embedding vector(768), -- nomic-embed-text-v2-moe (768-d)
  metadata JSONB DEFAULT '{}',
  importance FLOAT DEFAULT 0.5,
  source_ref TEXT, -- originating event: turn:N, tool:<call id>, message:<id>
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_memories_agent_id ON memories (agent_id);
CREATE INDEX idx_memories_type ON memories (memory_type);
CREATE INDEX idx_memories_agent_type ON memories (agent_id, memory_type);
CREATE INDEX idx_memories_created_at ON memories (created_at);
CREATE INDEX idx_memories_metadata_gin ON memories USING gin (metadata);

-- HNSW index for fast cosine search (tuned for ~128GB RAM hosts)
CREATE INDEX idx_memories_embedding ON memories 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Conversation snapshots
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL,
  messages JSONB NOT NULL DEFAULT '[]',
  summary TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_conversations_agent_id ON conversations (agent_id);
CREATE INDEX idx_conversations_created_at ON conversations (created_at);

CREATE UNIQUE INDEX uq_conversations_agent_id ON conversations (agent_id);

-- DPO gold pairs (offline role review)
CREATE TABLE dpo_gold_pairs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
  thread_id VARCHAR(256) NOT NULL DEFAULT '',
  user_text TEXT NOT NULL,
  rejected_reply TEXT NOT NULL,
  chosen_reply TEXT NOT NULL,
  critique TEXT NOT NULL DEFAULT '',
  role_failed BOOLEAN NOT NULL DEFAULT true,
  model VARCHAR(128) NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_dpo_gold_pairs_created_at ON dpo_gold_pairs (created_at);
CREATE INDEX idx_dpo_gold_pairs_agent_id ON dpo_gold_pairs (agent_id);

-- Canonical entities (synonym resolution for KG)
CREATE TABLE entities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
  canonical TEXT NOT NULL,
  aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
  kind TEXT,
  embedding vector(768),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX uq_entities_agent_canonical_lower ON entities (agent_id, (lower(canonical)));
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_entities_canonical_trgm ON entities USING gin (lower(canonical) gin_trgm_ops);
CREATE INDEX idx_entities_aliases_gin ON entities USING gin (aliases);
CREATE INDEX idx_entities_embedding ON entities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Knowledge graph edges (facts)
CREATE TABLE knowledge_graph (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
  subject VARCHAR(256) NOT NULL,
  predicate VARCHAR(256) NOT NULL,
  object TEXT NOT NULL,
  subject_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  object_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  triple_hash CHAR(32),
  embedding vector(768),
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_knowledge_graph_agent_id ON knowledge_graph (agent_id);
CREATE INDEX idx_kg_agent_subject ON knowledge_graph (agent_id, subject_id);
CREATE INDEX idx_knowledge_graph_created_at ON knowledge_graph (created_at);

CREATE UNIQUE INDEX uq_knowledge_graph_agent_triple_hash
  ON knowledge_graph (agent_id, triple_hash);

CREATE INDEX idx_knowledge_graph_embedding ON knowledge_graph
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- First-person self-reflections (agent diary; not third-person KG facts)
CREATE TABLE self_reflections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
  content TEXT NOT NULL,
  embedding vector(768),
  importance REAL DEFAULT 0.5,
  source_ref TEXT, -- originating event: turn:N, solitary:<thread>, outreach:<id>
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_self_reflections_agent_id ON self_reflections (agent_id);
CREATE INDEX idx_self_reflections_created_at ON self_reflections (created_at);

CREATE INDEX self_reflections_embedding_idx ON self_reflections
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Cryptographic identity
CREATE TABLE IF NOT EXISTS agent_identity (
    id SERIAL PRIMARY KEY,
    agent TEXT NOT NULL UNIQUE,
    public_key_pem TEXT NOT NULL,
    private_key_encrypted BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version INTEGER NOT NULL DEFAULT 1
);

-- Heart daemon vitals
CREATE TABLE IF NOT EXISTS vitals (
    tick_id BIGSERIAL PRIMARY KEY,
    tick_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    heart_pid INTEGER NOT NULL,
    body_metrics JSONB NOT NULL,
    phylactery_status TEXT NOT NULL
        CHECK (phylactery_status IN ('ok', 'degraded', 'fail')),
    signature TEXT
);

CREATE INDEX IF NOT EXISTS vitals_tick_at_idx ON vitals (tick_at DESC);

-- Signature column for self_reflections
ALTER TABLE self_reflections ADD COLUMN IF NOT EXISTS signature TEXT;

-- Source column for self_reflections (solitary cycle)
ALTER TABLE self_reflections ADD COLUMN IF NOT EXISTS source VARCHAR(32) NOT NULL DEFAULT 'reflect';

-- Agent activity — last live turn timestamp (time-sense / activity)
CREATE TABLE IF NOT EXISTS agent_activity (
  agent_id VARCHAR(64) PRIMARY KEY,
  last_turn_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Outbox — proactive outreach message queue
-- status: pending → sending → sent | expired (sending = claimed for Telegram)
-- footer_text: same ``→ turn …`` line as live Telegram (stamped after graph pass)
CREATE TABLE IF NOT EXISTS outbox (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
  created_at TIMESTAMPTZ DEFAULT now(),
  trigger_type VARCHAR(32) NOT NULL,
  draft_text TEXT NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'pending',
  sent_at TIMESTAMPTZ,
  footer_text TEXT,
  attempts INT NOT NULL DEFAULT 0,
  next_attempt_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbox_status_created ON outbox (status, created_at);

-- Mind Hub — unified cross-channel timeline (live, solitary, outreach)
CREATE TABLE IF NOT EXISTS mind_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL DEFAULT 'agent',
  kind VARCHAR(32) NOT NULL,
  channel VARCHAR(32),
  thread_id VARCHAR(128),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mind_events_agent_created ON mind_events (agent_id, created_at DESC);

-- Pulse — persistent scheduled intentions
CREATE TABLE IF NOT EXISTS intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id VARCHAR(64) NOT NULL,
  text TEXT NOT NULL,
  run_at TIMESTAMPTZ NOT NULL,
  recur_rule VARCHAR(64),
  status VARCHAR(16) NOT NULL DEFAULT 'pending',
  created_by VARCHAR(16) NOT NULL DEFAULT 'agent',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_run_at TIMESTAMPTZ,
  attempts INT NOT NULL DEFAULT 0,
  last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_intents_agent_status_run_at ON intents (agent_id, status, run_at);

-- Causal lessons — structured "changed X because Y"
CREATE TABLE IF NOT EXISTS causal_lessons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id TEXT NOT NULL,
  change_text TEXT NOT NULL,
  because_text TEXT NOT NULL,
  evidence_ids TEXT[] NOT NULL DEFAULT '{}',
  embedding vector(768),
  importance REAL NOT NULL DEFAULT 0.5,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_causal_lessons_agent_created
  ON causal_lessons (agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_causal_lessons_embedding
  ON causal_lessons
  USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_causal_lessons_importance
  ON causal_lessons (agent_id, importance DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_causal_lessons_agent_texts
  ON causal_lessons (agent_id, change_text, because_text);

-- Durable behavioral policy knobs (not diary / not identity)
CREATE TABLE IF NOT EXISTS policy_knobs (
  agent_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  lesson_id UUID NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (agent_id, key)
);

-- Note: LangGraph checkpoints are created by PostgresSaver.setup()
