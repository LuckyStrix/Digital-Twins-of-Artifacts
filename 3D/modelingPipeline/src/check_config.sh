#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROGRAM=$0

# script to check if dependencies are installed
print_red() {
    printf "\e[31m%s\e[0m" "$1"
}

print_green() {
    printf "\e[32m%s\e[0m" "$1"
}

status() {
    printf "%s: " "$PROGRAM"
    #printf "\e[32mstatus\e[0m"
    printf "$1" "${@:2}" >&2 
}

# system dependencies
#SD="python3 colmap gcc make tcc"
#SD="$1"
# python dependencies
#PD="numpy open3d"
#SD="$2"
# list of packages to install
ILIST=
PYPREFIX="python3-"

SYSTEM_DEPS=()
PYTHON_DEPS=()

mode="system"
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        mode="python"
        continue
    fi

    if [ "$mode" = "system" ]; then
        SYSTEM_DEPS+=("$arg")
    else
        PYTHON_DEPS+=("$arg")
    fi
done



status "SYSTEM DEPENDENCIES:\n"
# system dependencies
for p in "${SYSTEM_DEPS[@]}"; do
    status "%s: " "$p"
    IP=$(which "$p" 2>/dev/null || true)   # <-- prevents exit
    if [ -z "$IP" ]; then
        #printf "not installed!!!" "$p"
        print_red "not installed"
        printf "\n"
        ILIST="$ILIST $p"
        continue
    fi
    print_green "installed "
    printf "at %s\n" "$IP"
    #printf "%s: %s\n" "$p" "$IP"
done
status "PYTHON3 DEPENDENCIES:\n"
# python dependencies
for p in "${PYTHON_DEPS[@]}"; do
	status "python3 lib %s: " "$p"
    if ! python3 -c "import $p" >/dev/null 2>&1; then
        print_red "not installed"
        printf "\n"
        ILIST="$ILIST ${PYPREFIX}${p}"
        continue
    fi
    print_green "installed"
    printf "\n"

    #printf "at %s\n" "$IP"
    #printf "%s: %s\n" "$p" "$IP"
done

if [[ -n  $ILIST ]]; then
    if [ -r /etc/os-release ]; then
        . /etc/os-release
        if [ "$ID" = "debian" ] || [ "$ID_LIKE" = "debian" ]; then
            status "debian-based system detected\n"
            status "to install required dependencies, run: sudo apt-get install %s\n" "$(echo $ILIST)"
        else
            status "not Debian-based: %d" $ID
            status "you will need to find packages containing the required programs for your package manager" "$(echo $ILIST)"
        fi
    else
        status "/etc/os-release not found, cannot determine distro"
    fi
    exit 1;
else 
    exit 0
fi
