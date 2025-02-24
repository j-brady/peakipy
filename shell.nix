 let
   nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.11";
   pkgs = import nixpkgs { config = {}; overlays = []; };
 in

pkgs.mkShell {
  packages = with pkgs; [
    pkgs.stdenv.cc.cc.lib
    pkgs.python310
    pkgs.poetry
  ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  shellHook = ''
  poetry install 
  '';
}
