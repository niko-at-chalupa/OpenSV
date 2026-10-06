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
    stdenv.cc.cc.lib  # Provides libstdc++.so.6

    # Only needed if you also want to build the GUI editor (target: OpenSV).
    alsa-lib
    xorg.libX11
    xorg.libXext
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXcursor
    xorg.libXi

    (pkgs.python3.withPackages (ps: with ps; [ numpy ]))
  ];

  # Expose the C++ standard library to pip-installed wheels
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}:$LD_LIBRARY_PATH
  '';
}