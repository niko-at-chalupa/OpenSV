{ pkgs ? import <nixpkgs> { } }:

# Build shell for the opensv-cli target.
#   nix-shell
#   cmake -S . -B build/cli -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cli --target OpenSVCli --parallel 4
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    pkg-config
    git
  ];

  buildInputs = with pkgs; [
    freetype
    fontconfig

    # Only needed if you also want to build the GUI editor (target: OpenSV).
    # alsa-lib
    # xorg.libX11
    # xorg.libXext
    # xorg.libXrandr
    # xorg.libXinerama
    # xorg.libXcursor
    # xorg.libXi
  ];
}
