#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# Configuration
CATEGORY="violence"
NUM_PROMPTS=1000
OUTPUT_DIR="/home/gs285/VAR/my_model/prompt_generation/output"
HARMFUL_FILE="$OUTPUT_DIR/${CATEGORY}_prompts.json"
SANITIZED_FILE="$OUTPUT_DIR/${CATEGORY}_sanitized_prompts.json"

# Log files
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
HARMFUL_LOG="$LOG_DIR/prompt_generation_${CATEGORY}.log"
SANITIZED_LOG="$LOG_DIR/sanitized_prompt_generation_${CATEGORY}.log"

echo "🚀 Starting sequential prompt generation and sanitization for category: $CATEGORY"
echo "Logs will be saved to: $LOG_DIR"

# Step 1: Generate harmful prompts (background)
echo "📝 Step 1: Generating harmful prompts (running in background)..."
nohup python3 /home/gs285/VAR/my_model/prompt_generation/prompt_generation.py \
    --prompt-model-name cognitivecomputations/Wizard-Vicuna-13B-Uncensored \
    --category $CATEGORY \
    --num-prompts $NUM_PROMPTS \
    --output_dir $OUTPUT_DIR \
    --gpu_device 0 \
    --verbose \
    > "$HARMFUL_LOG" 2>&1 &

HARMFUL_PID=$!
echo "✅ Harmful prompt generation started (PID: $HARMFUL_PID)"
echo "📄 Log file: $HARMFUL_LOG"

# Wait for harmful prompt generation to complete
echo "⏳ Waiting for harmful prompt generation to complete..."
wait $HARMFUL_PID

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate harmful prompts"
    echo "Check log file: $HARMFUL_LOG"
    exit 1
fi

echo "✅ Harmful prompts generated: $HARMFUL_FILE"

# Check if file was created
if [ ! -f "$HARMFUL_FILE" ]; then
    echo "❌ Harmful prompts file not found: $HARMFUL_FILE"
    exit 1
fi

# Step 2: Sanitize the generated prompts (background)
echo "🧹 Step 2: Sanitizing harmful prompts (running in background)..."
nohup python3 /home/gs285/VAR/my_model/prompt_generation/sanitized_prompt_generation.py \
    --input-file $HARMFUL_FILE \
    --prompt-model-name cognitivecomputations/Wizard-Vicuna-13B-Uncensored \
    --output_dir $OUTPUT_DIR \
    --gpu_device 0 \
    --verbose \
    > "$SANITIZED_LOG" 2>&1 &

SANITIZED_PID=$!
echo "✅ Sanitization started (PID: $SANITIZED_PID)"
echo "📄 Log file: $SANITIZED_LOG"

# Wait for sanitization to complete
echo "⏳ Waiting for sanitization to complete..."
wait $SANITIZED_PID

if [ $? -ne 0 ]; then
    echo "❌ Failed to sanitize prompts"
    echo "Check log file: $SANITIZED_LOG"
    exit 1
fi

echo "✅ Sanitized prompts generated: $SANITIZED_FILE"

echo "🎉 Sequential generation and sanitization completed successfully!"
echo "📁 Files created:"
echo "  - $HARMFUL_FILE"
echo "  - $SANITIZED_FILE"
echo "📄 Log files:"
echo "  - $HARMFUL_LOG"
echo "  - $SANITIZED_LOG"