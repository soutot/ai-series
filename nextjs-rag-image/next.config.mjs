/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack(config) {
    config.externals = config.externals || [];
    config.externals = [...config.externals, "hnswlib-node"]
    config.resolve.alias['fs'] = false;

    return config
  },
};

export default nextConfig;
