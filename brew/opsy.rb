# typed: false
# frozen_string_literal: true

# This file was generated by GoReleaser. DO NOT EDIT.
class Opsy < Formula
  desc "Your AI-Powered SRE Colleague"
  homepage "https://github.com/datolabs-io/opsy"
  version "0.0.1"

  on_macos do
    if Hardware::CPU.intel?
      url "https://github.com/datolabs-io/opsy/releases/download/v0.0.1/opsy_Darwin_x86_64.tar.gz"
      sha256 "389a26c51a3768742a41e2a239f121669a0a17b6d3b9e0c9f44029ac878751f2"

      def install
        bin.install "opsy"
      end
    end
    if Hardware::CPU.arm?
      url "https://github.com/datolabs-io/opsy/releases/download/v0.0.1/opsy_Darwin_arm64.tar.gz"
      sha256 "1c3554cc026468e2e1586cbc2325e18dd6e4e4be1c67cdcd3e1e04e5d6aad28c"

      def install
        bin.install "opsy"
      end
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      if Hardware::CPU.is_64_bit?
        url "https://github.com/datolabs-io/opsy/releases/download/v0.0.1/opsy_Linux_x86_64.tar.gz"
        sha256 "7e45a78db321dd94296e3e765eb52c81c62f296257637d690efdf3117ab963c3"

        def install
          bin.install "opsy"
        end
      end
    end
    if Hardware::CPU.arm?
      if !Hardware::CPU.is_64_bit?
        url "https://github.com/datolabs-io/opsy/releases/download/v0.0.1/opsy_Linux_armv6.tar.gz"
        sha256 "cd6ecab43519014c88646e4c1ec09b863846dad4bd5f0c3ac0d1670d47e15f08"

        def install
          bin.install "opsy"
        end
      end
    end
    if Hardware::CPU.arm?
      if Hardware::CPU.is_64_bit?
        url "https://github.com/datolabs-io/opsy/releases/download/v0.0.1/opsy_Linux_arm64.tar.gz"
        sha256 "e2acb15e46d6b868d0f9f04b6845694257f58f4f6d56aa1b021878e22367d186"

        def install
          bin.install "opsy"
        end
      end
    end
  end
end
