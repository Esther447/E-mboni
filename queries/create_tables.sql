-- =============================================================
-- E-mboni PostgreSQL Schema
-- Paste this entire file into the VS Code PostgreSQL query tab
-- and press Run (F5). Run it only once.
-- =============================================================


-- -------------------------------------------------------------
-- STEP 1 — ENUM TYPES
-- -------------------------------------------------------------

CREATE TYPE role_enum         AS ENUM ('blind', 'guardian', 'admin');
CREATE TYPE language_enum     AS ENUM ('en', 'rw');
CREATE TYPE voice_speed_enum  AS ENUM ('Slow', 'Normal', 'Fast');
CREATE TYPE status_enum       AS ENUM ('active', 'inactive');
CREATE TYPE alert_level_enum  AS ENUM ('safe', 'warning', 'danger');
CREATE TYPE session_status_enum AS ENUM ('active', 'ended');


-- -------------------------------------------------------------
-- STEP 2 — USERS TABLE
-- -------------------------------------------------------------

CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255)        NOT NULL,
    phone           VARCHAR(20)         NOT NULL UNIQUE,
    password_hash   TEXT                NOT NULL,
    role            role_enum           NOT NULL,
    language        language_enum       NOT NULL DEFAULT 'en',
    voice_speed     voice_speed_enum    NOT NULL DEFAULT 'Normal',
    status          status_enum         NOT NULL DEFAULT 'active',
    guardian_id     INTEGER             REFERENCES users(id) ON DELETE SET NULL,
    emergency_phone VARCHAR(20),
    relationship    VARCHAR(100),
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_phone       ON users(phone);
CREATE INDEX idx_users_guardian_id ON users(guardian_id);


-- -------------------------------------------------------------
-- STEP 3 — ALERTS TABLE
-- -------------------------------------------------------------

CREATE TABLE alerts (
    id         SERIAL PRIMARY KEY,
    blind_id   INTEGER            NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message    TEXT               NOT NULL,
    level      alert_level_enum   NOT NULL DEFAULT 'warning',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alerts_blind_id ON alerts(blind_id);


-- -------------------------------------------------------------
-- STEP 4 — SESSIONS TABLE
-- -------------------------------------------------------------

CREATE TABLE sessions (
    id         SERIAL PRIMARY KEY,
    blind_id   INTEGER               NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ended_at   TIMESTAMP WITH TIME ZONE,
    status     session_status_enum   NOT NULL DEFAULT 'active'
);

CREATE INDEX idx_sessions_blind_id ON sessions(blind_id);
CREATE INDEX idx_sessions_status   ON sessions(status);


-- -------------------------------------------------------------
-- STEP 5 — SEED DEMO ACCOUNTS
--
--   Role     | Phone              | Password
--   ---------|--------------------|------------
--   admin    | +250 711 000 000   | admin123
--   guardian | +250 711 000 001   | guardian123
--   blind    | +250 711 000 002   | blind123
-- -------------------------------------------------------------

INSERT INTO users (name, phone, password_hash, role, language, voice_speed, status)
VALUES (
    'Admin',
    '+250780000000',
    '$2b$12$Dj6M1TWg2REoMGJRZaEc9ed8pd0FAezMvD.oQzGIcR291ANf3b3NW',
    'admin', 'en', 'Normal', 'active'
);

INSERT INTO users (name, phone, password_hash, role, language, voice_speed, status, relationship)
VALUES (
    'Sarah Kamau',
    '+250781000001',
    '$2b$12$kYVhKpEkRO76Kc9ixbxZeOSYTZNf4A5P49ZRE13KiofqXN2wQCs2m',
    'guardian', 'en', 'Normal', 'active', 'Mother'
);

INSERT INTO users (name, phone, password_hash, role, language, voice_speed, status, guardian_id, emergency_phone)
VALUES (
    'James Kamau',
    '+250781000002',
    '$2b$12$qMWpxPj0yljkKIVWozIb.eS57V0immj8qYjCUs8HgMi0/1PwZUJvi',
    'blind', 'en', 'Normal', 'active',
    (SELECT id FROM users WHERE phone = '+250781000001'),
    '+250780000000'
);


-- -------------------------------------------------------------
-- VERIFY — run this after to confirm everything was created
-- -------------------------------------------------------------

SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

SELECT id, name, phone, role FROM users ORDER BY id;
