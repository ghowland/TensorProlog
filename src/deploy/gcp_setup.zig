// ============================================================
// src/deploy/gcp_setup.zig
// ============================================================

const std_gcp = @import("std");

pub const GcpInstanceConfig = struct {
    project_id: [64]u8,
    project_id_len: i32,
    zone: [32]u8,
    zone_len: i32,
    machine_type: [32]u8,
    machine_type_len: i32,
    gpu_type: [32]u8,
    gpu_type_len: i32,
    gpu_count: i32,
    instance_name: [64]u8,
    instance_name_len: i32,
    disk_size_gb: i32,
    image_family: [32]u8,
    image_family_len: i32,
};

pub fn defaultGcpConfig() GcpInstanceConfig {
    var cfg = GcpInstanceConfig{
        .project_id = undefined,
        .project_id_len = 0,
        .zone = undefined,
        .zone_len = 0,
        .machine_type = undefined,
        .machine_type_len = 0,
        .gpu_type = undefined,
        .gpu_type_len = 0,
        .gpu_count = 1,
        .instance_name = undefined,
        .instance_name_len = 0,
        .disk_size_gb = 200,
        .image_family = undefined,
        .image_family_len = 0,
    };
    setField(&cfg.zone, &cfg.zone_len, "us-central1-a");
    setField(&cfg.machine_type, &cfg.machine_type_len, "n1-standard-8");
    setField(&cfg.gpu_type, &cfg.gpu_type_len, "nvidia-tesla-t4");
    setField(&cfg.image_family, &cfg.image_family_len, "ubuntu-2204-lts");
    return cfg;
}

fn setField(dest: []u8, dest_len: *i32, src: []const u8) void {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    dest_len.* = @intCast(n);
}

pub fn buildCreateCommand(config: *const GcpInstanceConfig, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "gcloud compute instances create ");
    pos += copyField(output[pos..], config.instance_name[0..@intCast(config.instance_name_len)]);
    pos += copyStr(output[pos..], " --project=");
    pos += copyField(output[pos..], config.project_id[0..@intCast(config.project_id_len)]);
    pos += copyStr(output[pos..], " --zone=");
    pos += copyField(output[pos..], config.zone[0..@intCast(config.zone_len)]);
    pos += copyStr(output[pos..], " --machine-type=");
    pos += copyField(output[pos..], config.machine_type[0..@intCast(config.machine_type_len)]);
    pos += copyStr(output[pos..], " --accelerator=type=");
    pos += copyField(output[pos..], config.gpu_type[0..@intCast(config.gpu_type_len)]);
    pos += copyStr(output[pos..], ",count=");
    pos += i32ToAsciiG(config.gpu_count, output[pos..]);
    pos += copyStr(output[pos..], " --boot-disk-size=");
    pos += i32ToAsciiG(config.disk_size_gb, output[pos..]);
    pos += copyStr(output[pos..], "GB");
    pos += copyStr(output[pos..], " --image-family=");
    pos += copyField(output[pos..], config.image_family[0..@intCast(config.image_family_len)]);
    pos += copyStr(output[pos..], " --image-project=ubuntu-os-cloud");
    pos += copyStr(output[pos..], " --maintenance-policy=TERMINATE");
    return @intCast(pos);
}

pub fn buildStartCommand(config: *const GcpInstanceConfig, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "gcloud compute instances start ");
    pos += copyField(output[pos..], config.instance_name[0..@intCast(config.instance_name_len)]);
    pos += copyStr(output[pos..], " --project=");
    pos += copyField(output[pos..], config.project_id[0..@intCast(config.project_id_len)]);
    pos += copyStr(output[pos..], " --zone=");
    pos += copyField(output[pos..], config.zone[0..@intCast(config.zone_len)]);
    return @intCast(pos);
}

pub fn buildStopCommand(config: *const GcpInstanceConfig, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "gcloud compute instances stop ");
    pos += copyField(output[pos..], config.instance_name[0..@intCast(config.instance_name_len)]);
    pos += copyStr(output[pos..], " --project=");
    pos += copyField(output[pos..], config.project_id[0..@intCast(config.project_id_len)]);
    pos += copyStr(output[pos..], " --zone=");
    pos += copyField(output[pos..], config.zone[0..@intCast(config.zone_len)]);
    return @intCast(pos);
}

pub fn buildDeleteCommand(config: *const GcpInstanceConfig, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "gcloud compute instances delete ");
    pos += copyField(output[pos..], config.instance_name[0..@intCast(config.instance_name_len)]);
    pos += copyStr(output[pos..], " --project=");
    pos += copyField(output[pos..], config.project_id[0..@intCast(config.project_id_len)]);
    pos += copyStr(output[pos..], " --zone=");
    pos += copyField(output[pos..], config.zone[0..@intCast(config.zone_len)]);
    pos += copyStr(output[pos..], " --quiet");
    return @intCast(pos);
}

pub fn buildSshCommand(config: *const GcpInstanceConfig, remote_cmd: []const u8, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "gcloud compute ssh ");
    pos += copyField(output[pos..], config.instance_name[0..@intCast(config.instance_name_len)]);
    pos += copyStr(output[pos..], " --project=");
    pos += copyField(output[pos..], config.project_id[0..@intCast(config.project_id_len)]);
    pos += copyStr(output[pos..], " --zone=");
    pos += copyField(output[pos..], config.zone[0..@intCast(config.zone_len)]);
    pos += copyStr(output[pos..], " --command=\"");
    pos += copyField(output[pos..], remote_cmd);
    pos += copyStr(output[pos..], "\"");
    return @intCast(pos);
}

fn copyStr(dest: []u8, src: []const u8) usize {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    return n;
}

fn copyField(dest: []u8, src: []const u8) usize {
    return copyStr(dest, src);
}

fn i32ToAsciiG(val: i32, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v: i64 = @intCast(val);
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf: [12]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    for (0..len) |i| {
        if (pos >= output.len) break;
        output[pos] = buf[len - 1 - i];
        pos += 1;
    }
    return pos;
}
