# shell.nix
{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone"; # choose your own name
  multiPkgs = pkgs: with pkgs; [ # choose your libraries
    libgcc
    binutils
    coreutils
  ];
  # exporting LIBRARY_PATH is essential so the libraries get found!
  profile = ''
    export LIBRARY_PATH=/usr/lib:/usr/lib64:$LIBRARY_PATH
    # export LIBRARY_PATH=${pkgs.libgcc}/lib # somethingl like this may also be necessary for certain libraries (change to your library e.g. pkgs.coreutils
  '';
  runScript = "bash";
}).env