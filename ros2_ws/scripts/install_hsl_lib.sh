#!/usr/bin/env bash
# ============================================================================
# install_hsl_lib.sh — Build and install the HSL MA57 sparse linear solver
# ============================================================================
#
# The RNTC-MPC framework uses IPOPT with the HSL MA57 linear solver, which
# requires a separate licence from the Numerical Algorithms Group (NAG):
#
#   https://licences.stfc.ac.uk/product/coin-hsl
#
# Once you have the source archive, place it in ThirdParty-HSL/ according to
# the instructions at https://github.com/coin-or-tools/ThirdParty-HSL
# and then run this script.
#
# Usage:
#   chmod +x scripts/install_hsl_lib.sh
#   ./scripts/install_hsl_lib.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "${SCRIPT_DIR}")"
HSL_DIR="${WS_DIR}/ThirdParty-HSL"

echo "=== HSL MA57 solver installer ==="
echo "Workspace: ${WS_DIR}"
echo "HSL source: ${HSL_DIR}"
echo

# Validate that the ThirdParty-HSL directory exists and contains configure
if [[ ! -d "${HSL_DIR}" ]]; then
    echo "ERROR: Directory not found: ${HSL_DIR}"
    echo
    echo "Please obtain the HSL source from:"
    echo "  https://licences.stfc.ac.uk/product/coin-hsl"
    echo "and place it in ${HSL_DIR} following the instructions at:"
    echo "  https://github.com/coin-or-tools/ThirdParty-HSL"
    exit 1
fi

if [[ ! -f "${HSL_DIR}/configure" ]]; then
    echo "ERROR: ${HSL_DIR}/configure not found."
    echo "The HSL source archive may not have been extracted correctly."
    echo "See: https://github.com/coin-or-tools/ThirdParty-HSL"
    exit 1
fi

cd "${HSL_DIR}"

echo "--- Running ./configure ---"
./configure

echo "--- Building (make) ---"
make -j"$(nproc)"

echo "--- Installing (sudo make install) ---"
sudo make install

echo
echo "=== HSL MA57 installed successfully ==="
echo "Shared library should now be at: /usr/local/lib/libcoinhsl.so"

# Verify the library is accessible
if ldconfig -p | grep -q libcoinhsl; then
    echo "Library confirmed in ld.so cache."
else
    echo "WARNING: libcoinhsl not found in ld.so cache."
    echo "You may need to run: sudo ldconfig"
    sudo ldconfig
fi

cd "${WS_DIR}"
