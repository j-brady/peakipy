 let
   nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-25.05";
   pkgs = import nixpkgs { config = {}; overlays = []; };
 in

pkgs.mkShell {
  buildInputs = with pkgs; [
    pkgs.stdenv.cc.cc.lib
    pkgs.python3
    pkgs.uv
    pkgs.openssl
    pkgs.pkg-config
    pkgs.libffi
  ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  shellHook = ''echo "uv version: $(uv --version)"
  '';
}
