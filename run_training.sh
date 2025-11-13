#!/bin/bash
# HP-VADE Training Launcher
# Convenient script for managing training on remote servers

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.7+"
    exit 1
fi

# Show menu
show_menu() {
    print_header "HP-VADE Training Manager"
    echo "1. Quick Test (2 epochs)"
    echo "2. Full Training (default settings)"
    echo "3. Full Training (custom settings)"
    echo "4. Monitor Training (one-time)"
    echo "5. Monitor Training (continuous)"
    echo "6. Start TensorBoard"
    echo "7. Check Training Status"
    echo "8. View Results"
    echo "9. Clean up old runs"
    echo "0. Exit"
    echo
    read -p "Select option [0-9]: " choice
}

# Quick test
quick_test() {
    print_header "Running Quick Test"
    python train_hp_vade.py --quick-test -y
    print_success "Quick test completed!"
}

# Full training with default settings
full_training_default() {
    print_header "Starting Full Training (Default Settings)"

    read -p "Run in background? [y/N]: " bg

    if [[ $bg =~ ^[Yy]$ ]]; then
        print_warning "Starting training in background..."
        nohup python train_hp_vade.py -y > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        PID=$!
        print_success "Training started with PID: $PID"
        print_warning "Log file: training_$(date +%Y%m%d_%H%M%S).log"
        print_warning "Monitor with: tail -f training_*.log"
    else
        python train_hp_vade.py -y
    fi
}

# Full training with custom settings
full_training_custom() {
    print_header "Full Training (Custom Settings)"

    echo "Enter custom parameters (or press Enter for default):"
    read -p "Learning rate [0.001]: " lr
    lr=${lr:-0.001}

    read -p "Batch size [128]: " bs
    bs=${bs:-128}

    read -p "Max epochs [100]: " epochs
    epochs=${epochs:-100}

    read -p "Experiment name [auto]: " exp_name

    read -p "Use CPU only? [y/N]: " use_cpu

    read -p "Run in background? [y/N]: " bg

    # Build command
    cmd="python train_hp_vade.py --lr $lr --batch-size $bs --max-epochs $epochs -y"

    if [[ ! -z "$exp_name" ]]; then
        cmd="$cmd --experiment-name $exp_name"
    fi

    if [[ $use_cpu =~ ^[Yy]$ ]]; then
        cmd="$cmd --cpu"
    fi

    print_warning "Command: $cmd"

    if [[ $bg =~ ^[Yy]$ ]]; then
        nohup $cmd > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        PID=$!
        print_success "Training started with PID: $PID"
    else
        $cmd
    fi
}

# Monitor training (one-time)
monitor_once() {
    print_header "Generating Training Report"
    python monitor_training.py
    print_success "Report generated in ./hp_vade_training/monitoring/"

    read -p "Open plots location? [y/N]: " open
    if [[ $open =~ ^[Yy]$ ]]; then
        ls -lh ./hp_vade_training/monitoring/
    fi
}

# Monitor training (continuous)
monitor_continuous() {
    print_header "Starting Continuous Monitoring"
    print_warning "Press Ctrl+C to stop"

    read -p "Update interval (seconds) [30]: " interval
    interval=${interval:-30}

    python monitor_training.py --watch --interval $interval
}

# Start TensorBoard
start_tensorboard() {
    print_header "Starting TensorBoard"

    read -p "Port [6006]: " port
    port=${port:-6006}

    read -p "Host [0.0.0.0 for remote access, localhost for local]: " host
    host=${host:-0.0.0.0}

    print_warning "Starting TensorBoard on $host:$port"
    print_warning "Access at: http://$(hostname -I | awk '{print $1}'):$port"
    print_warning "Or via SSH tunnel: ssh -L $port:localhost:$port user@server"

    tensorboard --logdir=./hp_vade_training/logs --host=$host --port=$port
}

# Check training status
check_status() {
    print_header "Training Status"

    # Check for running processes
    if pgrep -f "train_hp_vade.py" > /dev/null; then
        print_success "Training process is RUNNING"
        echo "Process IDs:"
        pgrep -af "train_hp_vade.py"

        # Show GPU usage if available
        if command -v nvidia-smi &> /dev/null; then
            echo -e "\nGPU Usage:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
        fi
    else
        print_warning "No training process found"
    fi

    # Check for checkpoints
    if [ -d "./hp_vade_training/checkpoints" ]; then
        echo -e "\nRecent checkpoints:"
        find ./hp_vade_training/checkpoints -name "*.ckpt" -type f -printf "%T@ %p\n" | sort -rn | head -5 | cut -d' ' -f2-
    fi

    # Check latest logs
    if [ -f "training_*.log" ]; then
        echo -e "\nLatest log entries:"
        tail -20 training_*.log 2>/dev/null | tail -10
    fi
}

# View results
view_results() {
    print_header "Training Results"

    if [ ! -d "./hp_vade_training/results" ]; then
        print_warning "No results directory found. Has training completed?"
        return
    fi

    # Show configuration
    if [ -f "./hp_vade_training/results/training_config.json" ]; then
        echo -e "\nTraining Configuration:"
        cat ./hp_vade_training/results/training_config.json
    fi

    # Show summary
    if [ -f "./hp_vade_training/results/training_summary.txt" ]; then
        echo -e "\nTraining Summary:"
        cat ./hp_vade_training/results/training_summary.txt
    fi

    # Show signature matrix info
    if [ -f "./hp_vade_training/results/signature_matrix.npy" ]; then
        print_success "Signature matrix saved"
        python -c "import numpy as np; S = np.load('./hp_vade_training/results/signature_matrix.npy'); print(f'Shape: {S.shape}, Min: {S.min():.6f}, Max: {S.max():.6f}, Mean: {S.mean():.6f}')"
    fi

    # List monitoring outputs
    if [ -d "./hp_vade_training/monitoring" ]; then
        echo -e "\nMonitoring outputs:"
        ls -lh ./hp_vade_training/monitoring/
    fi
}

# Clean up old runs
cleanup() {
    print_header "Cleanup Old Runs"
    print_warning "This will delete old checkpoints and logs"

    read -p "Keep last N experiments [3]: " keep
    keep=${keep:-3}

    read -p "Proceed? [y/N]: " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_warning "Cleanup cancelled"
        return
    fi

    # Cleanup checkpoints (keep last N)
    if [ -d "./hp_vade_training/checkpoints" ]; then
        echo "Cleaning up checkpoints..."
        cd ./hp_vade_training/checkpoints
        ls -t | tail -n +$((keep+1)) | xargs -r rm -rf
        cd - > /dev/null
        print_success "Kept $keep most recent checkpoint directories"
    fi

    # Cleanup old log files
    find . -name "training_*.log" -mtime +7 -delete 2>/dev/null
    print_success "Deleted log files older than 7 days"
}

# Main loop
while true; do
    show_menu

    case $choice in
        1) quick_test ;;
        2) full_training_default ;;
        3) full_training_custom ;;
        4) monitor_once ;;
        5) monitor_continuous ;;
        6) start_tensorboard ;;
        7) check_status ;;
        8) view_results ;;
        9) cleanup ;;
        0)
            print_success "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option. Please select 0-9."
            ;;
    esac

    echo
    read -p "Press Enter to continue..."
done
