DROP TABLE IF EXISTS "users";
DROP TABLE IF EXISTS "groups";
DROP TABLE IF EXISTS "messages";
DROP TABLE IF EXISTS "credits";
DROP TABLE IF EXISTS "tags";
DROP TABLE IF EXISTS "group_tags";

DROP TYPE IF EXISTS "message_status";

CREATE TYPE "message_status" AS ENUM ('pending', 'success', 'failed');

CREATE TABLE "users" (
	"id" UUID UNIQUE,
	"username" VARCHAR(50) NOT NULL,
	-- Hashed password for security
	"password_hash" VARCHAR(255) NOT NULL,
	"associate" VARCHAR(255),
	"last_login" TIMESTAMP,
	"phone_no" VARCHAR(20), -- Changed from NUMERIC to handle international formats
	"device_flag" INTEGER,
	"created_at" TIMESTAMP DEFAULT now(),
	"updated_at" TIMESTAMP DEFAULT now(),
	PRIMARY KEY("id")
);

CREATE INDEX "users_username_idx"
ON "users" ("username");

CREATE TABLE "groups" (
	"id" UUID UNIQUE,
	"name" VARCHAR(100) NOT NULL,
	-- WhatsApp group identifier
	"whatsapp_group_id" VARCHAR(100) UNIQUE,
	"created_at" TIMESTAMP DEFAULT now(),
	"updated_at" TIMESTAMP DEFAULT now(),
	PRIMARY KEY("id")
);COMMENT ON COLUMN groups.whatsapp_group_id IS 'WhatsApp group identifier';

CREATE INDEX "groups_whatsapp_group_id_idx"
ON "groups" ("whatsapp_group_id");

CREATE INDEX "groups_name_idx"
ON "groups" ("name");

CREATE TABLE "messages" (
	"id" UUID UNIQUE,
	-- WhatsApp message identifier
	"whatsapp_message_id" VARCHAR(100) UNIQUE,
	"group_id" UUID NOT NULL, -- Changed from INTEGER to UUID
	"sender_id" UUID NOT NULL, -- Changed from INTEGER to UUID
	-- Actual message content
	"content" TEXT NOT NULL,
	-- URL for media files
	"media_url" VARCHAR(500),
	-- pending, sent, delivered, read, failed
	"status" VARCHAR(20) DEFAULT "pending" CHECK (status IN ("pending", "success", "failed")),
	-- For scheduled messages
	"scheduled_at" TIMESTAMP,
	"sent_at" TIMESTAMP,
	"delivered_at" TIMESTAMP,
	"created_at" TIMESTAMP DEFAULT now(),
	"updated_at" TIMESTAMP DEFAULT now(),
	PRIMARY KEY("id")
);

CREATE INDEX "messages_whatsapp_message_id_idx"
ON "messages" ("whatsapp_message_id");

CREATE INDEX "messages_content_idx"
ON "messages" ("content");

CREATE INDEX "messages_group_status_idx"
ON "messages" ("group_id", "status");

CREATE INDEX "messages_group_created_idx"
ON "messages" ("group_id", "created_at" DESC);

CREATE INDEX "messages_sender_created_idx"
ON "messages" ("sender_id", "created_at" DESC);

CREATE TABLE "credits" (
	"user_id" UUID,
	"limit" INTEGER,
	"left" INTEGER,
	"updated_at" TIMESTAMP DEFAULT now(),
	PRIMARY KEY("user_id")
);

CREATE INDEX "credits_user_id_idx"
ON "credits" ("user_id");

CREATE TABLE "tags" (
	"id" UUID NOT NULL UNIQUE,
	"name" VARCHAR(255) UNIQUE,
	"created_at" TIMESTAMP DEFAULT now(),
	PRIMARY KEY("id")
);

CREATE TABLE "group_tags" (
	"tag_id" UUID NOT NULL,
	"group_id" UUID NOT NULL,
	"created_at" TIMESTAMP DEFAULT now(),
	PRIMARY KEY("tag_id", "group_id")
);

CREATE INDEX "group_tags_group_id_idx"
ON "group_tags" ("group_id");

CREATE INDEX "group_tags_tag_id_idx" 
ON "group_tags" ("tag_id");

ALTER TABLE "messages"
ADD FOREIGN KEY("group_id") REFERENCES "groups"("id")
ON UPDATE NO ACTION ON DELETE CASCADE;

ALTER TABLE "messages"
ADD FOREIGN KEY("sender_id") REFERENCES "users"("id")
ON UPDATE NO ACTION ON DELETE CASCADE;

ALTER TABLE "credits"
ADD FOREIGN KEY("user_id") REFERENCES "users"("id")
ON UPDATE NO ACTION ON DELETE CASCADE;

ALTER TABLE "group_tags"
ADD FOREIGN KEY("group_id") REFERENCES "groups"("id")
ON UPDATE NO ACTION ON DELETE CASCADE;

ALTER TABLE "group_tags"
ADD FOREIGN KEY("tag_id") REFERENCES "tags"("id")
ON UPDATE NO ACTION ON DELETE CASCADE;