// Copyright (c) 2023 Gustavo C. Viegas. All rights reserved.

const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;

pub fn Vec2(comptime T: type) type {
    return VecN(2, T);
}

pub fn Vec3(comptime T: type) type {
    return VecN(3, T);
}

pub fn Vec4(comptime T: type) type {
    return VecN(4, T);
}

fn VecN(comptime n: comptime_int, comptime T: type) type {
    if (n <= 1) @compileError("VecN requires n > 1");
    return struct {
        const Self = @This();

        /// Note that this is not the same thing as `[n]T`.
        /// One should convert it to array when the layout matters.
        v: @Vector(n, T),

        pub fn add(self: Self, other: Self) Self {
            return Self{ .v = self.v + other.v };
        }

        pub fn sub(self: Self, other: Self) Self {
            return Self{ .v = self.v - other.v };
        }

        pub fn scale(self: Self, scalar: T) Self {
            return Self{ .v = self.v * @splat(n, scalar) };
        }

        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, self.v * other.v);
        }

        pub fn length(self: Self) T {
            return @sqrt(self.dot(self));
        }

        /// `self.length()` must not be zero.
        pub fn normalize(self: Self) Self {
            return self.scale(1 / self.length());
        }

        /// Works on vectors containing 3 or more elements.
        /// Elements in the range [3..], if present, will be ignored.
        pub fn cross(self: Self, other: Self) VecN(3, T) {
            if (n < 3) @compileError("cross product requires at least 3 elements");

            const mask1 = @Vector(3, i32){ 1, 2, 0 };
            const mask2 = @Vector(3, i32){ 2, 0, 1 };

            const a = @shuffle(T, self.v, undefined, mask1) *
                @shuffle(T, other.v, undefined, mask2);

            const b = @shuffle(T, other.v, undefined, mask1) *
                @shuffle(T, self.v, undefined, mask2);

            return VecN(3, T){ .v = a - b };
        }

        /// The last element is assumed to be the real part.
        pub fn mulQuaternion(self: VecN(4, T), other: VecN(4, T)) VecN(4, T) {
            const imag_1 = @shuffle(T, self.v, undefined, @Vector(3, i32){ 0, 1, 2 });
            const real_1 = @splat(3, self.v[3]);
            const imag_2 = @shuffle(T, other.v, undefined, @Vector(3, i32){ 0, 1, 2 });
            const real_2 = @splat(3, other.v[3]);

            const cross_1_2 = (VecN(3, T){ .v = imag_1 }).cross(VecN(3, T){ .v = imag_2 }).v;
            const dot_1_2 = (VecN(3, T){ .v = imag_1 }).dot(VecN(3, T){ .v = imag_2 });

            const imag = imag_1 * real_2 + imag_2 * real_1 + cross_1_2;
            // TODO: Maybe use scalar here and construct a @Vector later.
            const real = real_1 * real_2 - @splat(3, dot_1_2);

            return VecN(4, T){
                .v = @shuffle(T, imag, real, @Vector(4, i32){ 0, 1, 2, -1 }),
            };
        }

        /// New elements are set to zero.
        pub fn resize(self: Self, comptime new_n: comptime_int) VecN(new_n, T) {
            switch (new_n) {
                0...1 => @compileError("unreachable"),
                n => return self,
                else => {
                    const mask = comptime x: {
                        var m: @Vector(new_n, T) = undefined;
                        for (0..@min(n, new_n)) |i| {
                            m[i] = i;
                        }
                        if (new_n > n) {
                            for (n..new_n) |i| {
                                m[i] = -1;
                            }
                        }
                        break :x m;
                    };
                    return VecN(new_n, T){ .v = @shuffle(T, self.v, @Vector(1, T){0}, mask) };
                },
            }
        }
    };
}

test "linear.VecN.dot" {
    const v = Vec3(f32){ .v = .{ 0, 0.7071068, 0.7071068 } };
    const u = Vec3(f32){ .v = -v.v };
    try expectEqual(@round(v.dot(v)), 1);
    try expectEqual(@round(u.dot(u)), 1);
    try expectEqual(@round(v.dot(u)), -1);
}

test "linear.VecN.length" {
    const v2 = Vec2(f64){ .v = .{ 4, 3 } };
    const v3 = Vec3(f32){ .v = .{ 4, 0, 3 } };
    const v4 = Vec4(f16){ .v = .{ 0, 3, 0, 4 } };
    try expectEqual(v2.length(), 5);
    try expectEqual(v3.length(), 5);
    try expectEqual(v4.length(), 5);
}

test "linear.VecN.normalize" {
    const v = Vec3(f32){ .v = .{ 3, 0, 4 } };
    const vn = v.normalize();
    try expectEqual(vn.v, @TypeOf(vn.v){ 0.6, 0, 0.8 });
    try expectEqual(vn.length(), 1);
}

test "linear.VecN.cross" {
    const v = Vec3(f32){ .v = .{ 0, 0, 1 } };
    const u = Vec3(f32){ .v = .{ 0, 1, 0 } };
    var w = v.cross(u);
    try expectEqual(w.v, @TypeOf(w.v){ -1, 0, 0 });
    w = u.cross(v);
    try expectEqual(w.v, @TypeOf(w.v){ 1, 0, 0 });
    w = v.cross(v);
    try expectEqual(w.v, @TypeOf(w.v){ 0, 0, 0 });
    w = v.cross(v.scale(-1));
    try expectEqual(w.v, @TypeOf(w.v){ 0, 0, 0 });
}

test "linear.Vec4.mulQuaternion" {
    {
        const q = Vec4(f32){ .v = .{ 0, 0, 0, 1 } };
        const u = Vec4(f32){ .v = .{ 0.7071068, 0, -0.7071068, 1 } };
        const p = q.mulQuaternion(u);
        try expectEqual(p.v, u.v);
    }
    {
        const pi = 3.141593;
        const q = Vec4(f32){ .v = .{ 0, 0, -1, pi } };
        const u = Vec4(f32){ .v = .{ 1, 0, 0, pi } };
        const p = q.mulQuaternion(u);
        try expectEqual(p.v[0], pi);
        try expectEqual(p.v[1], -1);
        try expectEqual(p.v[2], -pi);
        try expect(@fabs(p.v[3] - pi * pi) < 0.000001);
    }
}

test "linear.VecN.resize" {
    const v = init4(f64, 0.1, 0.2, -0.3, -0.4);
    const u = v.resize(8);
    for (0..4) |i| {
        try expectEqual(v.v[i], u.v[i]);
        try expectEqual(u.v[i + 4], 0);
    }
    const w = v.resize(3);
    try expectEqual(w.v, @Vector(3, f64){ 0.1, 0.2, -0.3 });
    const s = v.resize(2);
    try expectEqual(s.v, @Vector(2, f64){ 0.1, 0.2 });
    const t = s.resize(4);
    try expectEqual(t.v, @Vector(4, f64){ 0.1, 0.2, 0, 0 });
    try expectEqual(w.resize(2).resize(3).resize(4).v, t.v);
    try expectEqual(v.resize(4).v, v.v);
}

pub fn Vec2x2(comptime T: type) type {
    return VecNxN(2, T);
}

pub fn Vec3x3(comptime T: type) type {
    return VecNxN(3, T);
}

pub fn Vec4x4(comptime T: type) type {
    return VecNxN(4, T);
}

/// Column-major.
fn VecNxN(comptime n: comptime_int, comptime T: type) type {
    if (n <= 1) @compileError("VecNxN requires n > 1");
    return struct {
        const Self = @This();

        v: @Vector(n * n, T),

        pub fn identity(self: *Self) void {
            self.v = @splat(n * n, @as(T, 0));
            for (0..n) |i| {
                self.v[i * n + i] = 1;
            }
        }

        pub fn add(self: Self, other: Self) Self {
            return Self{ .v = self.v + other.v };
        }

        pub fn sub(self: Self, other: Self) Self {
            return Self{ .v = self.v - other.v };
        }

        pub fn mul(self: Self, other: Self) Self {
            const row_mask = comptime x: {
                var m: [n]@Vector(n, i32) = undefined;
                for (0..n) |i| {
                    for (0..n) |j| {
                        m[i][j] = i + n * j;
                    }
                }
                break :x m;
            };

            const col_mask = comptime x: {
                var m: [n]@Vector(n, i32) = undefined;
                for (0..n) |i| {
                    for (0..n) |j| {
                        m[i][j] = i * n + j;
                    }
                }
                break :x m;
            };

            var v: @TypeOf(self.v) = undefined;

            inline for (0..n) |i| {
                inline for (0..n) |j| {
                    const row = @shuffle(T, self.v, undefined, row_mask[j]);
                    const col = @shuffle(T, other.v, undefined, col_mask[i]);
                    v[i * n + j] = @reduce(.Add, row * col);
                }
            }

            return Self{ .v = v };
        }

        pub fn mulVecN(self: Self, vecN: VecN(n, T)) VecN(n, T) {
            const row_mask = comptime x: {
                var m: [n]@Vector(n, i32) = undefined;
                for (0..n) |i| {
                    for (0..n) |j| {
                        m[i][j] = i + n * j;
                    }
                }
                break :x m;
            };

            var v: @TypeOf(vecN.v) = undefined;

            inline for (0..n) |i| {
                const row = @shuffle(T, self.v, undefined, row_mask[i]);
                v[i] = @reduce(.Add, row * vecN.v);
            }

            return VecN(n, T){ .v = v };
        }

        pub fn transpose(self: Self) Self {
            const mask = comptime x: {
                var m: @Vector(n * n, i32) = undefined;
                for (0..n) |i| {
                    for (0..n) |j| {
                        m[i * n + j] = i + n * j;
                    }
                }
                break :x m;
            };

            return Self{ .v = @shuffle(T, self.v, undefined, mask) };
        }

        pub fn det(self: Self) T {
            switch (n) {
                2 => return self.v[0] * self.v[3] - self.v[1] * self.v[2],
                3 => {
                    const m00 = self.v[0];
                    const m01 = self.v[1];
                    const m02 = self.v[2];
                    const m10 = self.v[3];
                    const m11 = self.v[4];
                    const m12 = self.v[5];
                    const m20 = self.v[6];
                    const m21 = self.v[7];
                    const m22 = self.v[8];
                    return m00 * (m11 * m22 - m12 * m21) -
                        m01 * (m10 * m22 - m12 * m20) +
                        m02 * (m10 * m21 - m11 * m20);
                },
                4 => {
                    const m00 = self.v[0];
                    const m01 = self.v[1];
                    const m02 = self.v[2];
                    const m03 = self.v[3];
                    const m10 = self.v[4];
                    const m11 = self.v[5];
                    const m12 = self.v[6];
                    const m13 = self.v[7];
                    const m20 = self.v[8];
                    const m21 = self.v[9];
                    const m22 = self.v[10];
                    const m23 = self.v[11];
                    const m30 = self.v[12];
                    const m31 = self.v[13];
                    const m32 = self.v[14];
                    const m33 = self.v[15];
                    return (m00 * m11 - m01 * m10) * (m22 * m33 - m23 * m32) -
                        (m00 * m12 - m02 * m10) * (m21 * m33 - m23 * m31) +
                        (m00 * m13 - m03 * m10) * (m21 * m32 - m22 * m31) +
                        (m01 * m12 - m02 * m11) * (m20 * m33 - m23 * m30) -
                        (m01 * m13 - m03 * m11) * (m20 * m32 - m22 * m30) +
                        (m02 * m13 - m03 * m12) * (m20 * m31 - m21 * m30);
                },
                else => @compileError("determinant not implemented for n > 4"),
            }
        }

        // TODO: Check that T is floating-point.
        pub fn invert(self: Self) Self {
            switch (n) {
                2 => {
                    const m00 = self.v[0];
                    const m01 = self.v[1];
                    const m10 = self.v[2];
                    const m11 = self.v[3];
                    const inv_det = 1 / (m00 * m11 - m01 * m10);
                    return Self{ .v = @splat(n * n, inv_det) * @Vector(n * n, T){
                        m11,  m01,
                        -m10, m00,
                    } };
                },
                3 => {
                    const m00 = self.v[0];
                    const m01 = self.v[1];
                    const m02 = self.v[2];
                    const m10 = self.v[3];
                    const m11 = self.v[4];
                    const m12 = self.v[5];
                    const m20 = self.v[6];
                    const m21 = self.v[7];
                    const m22 = self.v[8];
                    const s0 = m11 * m22 - m12 * m21;
                    const s1 = m10 * m22 - m12 * m20;
                    const s2 = m10 * m21 - m11 * m20;
                    const inv_det = 1 / (m00 * s0 - m01 * s1 + m02 * s2);
                    return Self{ .v = @splat(n * n, inv_det) * @Vector(n * n, T){
                        s0,
                        -(m01 * m22 - m02 * m21),
                        m01 * m12 - m02 * m11,

                        -s1,
                        m00 * m22 - m02 * m20,
                        -(m00 * m12 - m02 * m10),

                        s2,
                        -(m00 * m21 - m01 * m20),
                        m00 * m11 - m01 * m10,
                    } };
                },
                4 => {
                    const m00 = self.v[0];
                    const m01 = self.v[1];
                    const m02 = self.v[2];
                    const m03 = self.v[3];
                    const m10 = self.v[4];
                    const m11 = self.v[5];
                    const m12 = self.v[6];
                    const m13 = self.v[7];
                    const m20 = self.v[8];
                    const m21 = self.v[9];
                    const m22 = self.v[10];
                    const m23 = self.v[11];
                    const m30 = self.v[12];
                    const m31 = self.v[13];
                    const m32 = self.v[14];
                    const m33 = self.v[15];
                    const s0 = m00 * m11 - m01 * m10;
                    const s1 = m00 * m12 - m02 * m10;
                    const s2 = m00 * m13 - m03 * m10;
                    const s3 = m01 * m12 - m02 * m11;
                    const s4 = m01 * m13 - m03 * m11;
                    const s5 = m02 * m13 - m03 * m12;
                    const c0 = m20 * m31 - m21 * m30;
                    const c1 = m20 * m32 - m22 * m30;
                    const c2 = m20 * m33 - m23 * m30;
                    const c3 = m21 * m32 - m22 * m31;
                    const c4 = m21 * m33 - m23 * m31;
                    const c5 = m22 * m33 - m23 * m32;
                    const inv_det = 1 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
                    return Self{ .v = @splat(n * n, inv_det) * @Vector(n * n, T){
                        c5 * m11 - c4 * m12 + c3 * m13,
                        -c5 * m01 + c4 * m02 - c3 * m03,
                        s5 * m31 - s4 * m32 + s3 * m33,
                        -s5 * m21 + s4 * m22 - s3 * m23,

                        -c5 * m10 + c2 * m12 - c1 * m13,
                        c5 * m00 - c2 * m02 + c1 * m03,
                        -s5 * m30 + s2 * m32 - s1 * m33,
                        s5 * m20 - s2 * m22 + s1 * m23,

                        c4 * m10 - c2 * m11 + c0 * m13,
                        -c4 * m00 + c2 * m01 - c0 * m03,
                        s4 * m30 - s2 * m31 + s0 * m33,
                        -s4 * m20 + s2 * m21 - s0 * m23,

                        -c3 * m10 + c1 * m11 - c0 * m12,
                        c3 * m00 - c1 * m01 + c0 * m02,
                        -s3 * m30 + s1 * m31 - s0 * m32,
                        s3 * m20 - s1 * m21 + s0 * m22,
                    } };
                },
                else => @compileError("inversion not implemented for n > 4"),
            }
        }

        pub fn upperLeft(self: VecNxN(4, T)) VecNxN(3, T) {
            const mask = @Vector(9, i32){ 0, 1, 2, 4, 5, 6, 8, 9, 10 };
            return VecNxN(3, T){ .v = @shuffle(T, self.v, undefined, mask) };
        }

        pub fn columnAt(self: Self, comptime index: comptime_int) VecN(n, T) {
            const mask = comptime x: {
                var m: @Vector(n, i32) = undefined;
                for (0..n) |i| {
                    m[i] = index * n + i;
                }
                break :x m;
            };
            return VecN(n, T){ .v = @shuffle(T, self.v, undefined, mask) };
        }

        pub fn rowAt(self: Self, comptime index: comptime_int) VecN(n, T) {
            const mask = comptime x: {
                var m: @Vector(n, i32) = undefined;
                for (0..n) |i| {
                    m[i] = index + i * n;
                }
                break :x m;
            };
            return VecN(n, T){ .v = @shuffle(T, self.v, undefined, mask) };
        }
    };
}

test "linear.VecNxN.mul" {
    {
        const m = Vec2x2(i32){ .v = .{
            2, 3,
            4, 5,
        } };
        const n = Vec2x2(i32){ .v = .{
            2, 1,
            1, 2,
        } };
        var o = m.mul(n);
        try expectEqual(o.v, @TypeOf(o.v){ 8, 11, 10, 13 });
        o = n.mul(m);
        try expectEqual(o.v, @TypeOf(o.v){ 7, 8, 13, 14 });
        o.identity();
        var p = m.mul(o);
        try expectEqual(p.v, m.v);
        p = o.mul(n);
        try expectEqual(p.v, n.v);
    }
    {
        const m = Vec3x3(i32){ .v = .{
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
        } };
        const n = Vec3x3(i32){ .v = .{
            0, 1, 0,
            0, 0, 1,
            1, 0, 0,
        } };
        var o = m.mul(n);
        try expectEqual(
            o.v,
            @shuffle(i32, m.v, undefined, @Vector(9, i32){ 3, 4, 5, 6, 7, 8, 0, 1, 2 }),
        );
        o = o.mul(n);
        try expectEqual(
            o.v,
            @shuffle(i32, m.v, undefined, @Vector(9, i32){ 6, 7, 8, 0, 1, 2, 3, 4, 5 }),
        );
        o = o.mul(n);
        try expectEqual(o.v, m.v);
    }
}

test "linear.VecNxN.mulVecN" {
    var m = Vec3x3(i32){ .v = .{
        2, 0, 1,
        1, 3, 2,
        4, 2, 3,
    } };
    const v = Vec3(i32){ .v = .{ -1, 0, 1 } };
    var u = m.mulVecN(v);
    try expectEqual(u.v, @TypeOf(u.v){ 2, 2, 2 });
    m.identity();
    u = m.mulVecN(u);
    try expectEqual(u.v, @TypeOf(u.v){ 2, 2, 2 });
    u = m.mulVecN(v);
    try expectEqual(u.v, v.v);
}

test "linear.VecNxN.transpose" {
    {
        const m = Vec3x3(f64){ .v = .{
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
        } };
        const n = m.transpose();
        try expectEqual(
            n.v,
            @shuffle(f64, m.v, undefined, @Vector(9, i32){ 0, 3, 6, 1, 4, 7, 2, 5, 8 }),
        );
        const o = n.transpose();
        try expectEqual(o.v, m.v);
    }
    {
        var m: Vec4x4(i16) = undefined;
        m.identity();
        const n = m.transpose();
        try expectEqual(n.v, m.v);
        const o = n.transpose();
        try expectEqual(o.v, n.v);
    }
}

test "linear.VecNxN.det" {
    {
        const m = Vec2x2(f32){ .v = .{
            1, -1,
            2, -0.5,
        } };
        const det = m.det();
        try expectEqual(det, m.v[0] * m.v[3] - m.v[1] * m.v[2]);
    }
    {
        const m = Vec3x3(f32){ .v = .{
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        } };
        const det = m.det();
        try expectEqual(det, 0);
    }
    {
        const m = Vec4x4(f32){ .v = .{
            0.5, 0,  0, 0,
            0,   -2, 0, 0,
            0,   0,  6, 0,
            0,   0,  0, 1.5,
        } };
        const det = m.det();
        try expectEqual(det, m.v[0] * m.v[5] * m.v[10] * m.v[15]);
    }
}

test "linear.VecNxN.invert" {
    {
        const m = Vec2x2(f32){ .v = .{
            12, 0,
            -1, 4,
        } };
        const im = m.invert();
        const ii = im.invert();
        const n = m.mul(im);
        try expectEqual(ii.v, m.v);
        try expectEqual(n.v, @Vector(4, f32){ 1, 0, 0, 1 });
    }
    {
        const m = Vec3x3(f32){ .v = .{
            1, 0, 0,
            0, 1, 0,
            7, 8, 9,
        } };
        const im = m.invert();
        const ii = im.invert();
        const n = m.mul(im);
        try expectEqual(ii.v, m.v);
        try expectEqual(n.v, @Vector(9, f32){ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    }
    {
        const m = Vec4x4(f32){ .v = .{
            -2, 0,   0,  0,
            0,  -34, 0,  1,
            0,  0,   -1, 2,
            0,  1,   2,  -16,
        } };
        const im = m.invert();
        const ii = im.invert();
        const n = m.mul(im);
        try expectEqual(@round(ii.v), m.v);
        // XXX: Negative zero.
        try expectEqual(
            n.v,
            @Vector(16, f32){ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
        );
    }
}

test "linear.Vec4x4.upperLeft" {
    const m = init4x4(f32, 0.1, 0.2, 0.3, 1, 0.4, 0.5, 0.6, 2, 0.7, 0.8, 0.9, 3, 4, 5, 6, 7)
        .upperLeft();
    try expectEqual(m.v, @Vector(9, f32){ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });
}

test "linear.VecNxN.columnAt/rowAt" {
    const m = init3x3(i8, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    const n = m.transpose();
    inline for (0..3, [_]@Vector(3, i8){ .{ 9, 8, 7 }, .{ 6, 5, 4 }, .{ 3, 2, 1 } }) |i, x| {
        try expectEqual(x, m.columnAt(i).v);
        try expectEqual(x, n.rowAt(i).v);
    }
}

pub fn init2(comptime T: type, v0: T, v1: T) Vec2(T) {
    return Vec2(T){ .v = .{ v0, v1 } };
}

pub fn init3(comptime T: type, v0: T, v1: T, v2: T) Vec3(T) {
    return Vec3(T){ .v = .{ v0, v1, v2 } };
}

pub fn init4(comptime T: type, v0: T, v1: T, v2: T, v3: T) Vec4(T) {
    return Vec4(T){ .v = .{ v0, v1, v2, v3 } };
}

pub fn init2x2(
    comptime T: type,
    c0_r0: T,
    c0_r1: T,
    c1_r0: T,
    c1_r1: T,
) Vec2x2(T) {
    return Vec2x2(T){ .v = .{
        c0_r0, c0_r1,
        c1_r0, c1_r1,
    } };
}

pub fn init3x3(
    comptime T: type,
    c0_r0: T,
    c0_r1: T,
    c0_r2: T,
    c1_r0: T,
    c1_r1: T,
    c1_r2: T,
    c2_r0: T,
    c2_r1: T,
    c2_r2: T,
) Vec3x3(T) {
    return Vec3x3(T){ .v = .{
        c0_r0, c0_r1, c0_r2,
        c1_r0, c1_r1, c1_r2,
        c2_r0, c2_r1, c2_r2,
    } };
}

pub fn init4x4(
    comptime T: type,
    c0_r0: T,
    c0_r1: T,
    c0_r2: T,
    c0_r3: T,
    c1_r0: T,
    c1_r1: T,
    c1_r2: T,
    c1_r3: T,
    c2_r0: T,
    c2_r1: T,
    c2_r2: T,
    c2_r3: T,
    c3_r0: T,
    c3_r1: T,
    c3_r2: T,
    c3_r3: T,
) Vec4x4(T) {
    return Vec4x4(T){ .v = .{
        c0_r0, c0_r1, c0_r2, c0_r3,
        c1_r0, c1_r1, c1_r2, c1_r3,
        c2_r0, c2_r1, c2_r2, c2_r3,
        c3_r0, c3_r1, c3_r2, c3_r3,
    } };
}

test "linear.init*" {
    {
        const v = init2(u32, 100, 200);
        try expectEqual(v.v, @Vector(2, u32){ 100, 200 });
    }
    {
        const v = init3(i16, 32767, -1, -32768);
        try expectEqual(v.v, @Vector(3, i16){ 32767, -1, -32768 });
    }
    {
        const v = init4(f64, -0.125, 2.5, -66.1, 0.001);
        try expectEqual(v.v, @Vector(4, f64){ -0.125, 2.5, -66.1, 0.001 });
    }
    {
        const m = init2x2(u8, 1, 2, 3, 4);
        try expectEqual(m.v, @Vector(4, u8){ 1, 2, 3, 4 });
    }
    {
        const m = init3x3(i64, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        try expectEqual(m.v, @Vector(9, i64){ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
    }
    {
        const m = init4x4(f32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        try expectEqual(m.v, @Vector(16, f32){ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
    }
}

pub fn zero2(comptime T: type) Vec2(T) {
    return Vec2(T){ .v = .{ 0, 0 } };
}

pub fn zero3(comptime T: type) Vec3(T) {
    return Vec3(T){ .v = .{ 0, 0, 0 } };
}

pub fn zero4(comptime T: type) Vec4(T) {
    return Vec4(T){ .v = .{ 0, 0, 0, 0 } };
}

test "linear.zero*" {
    {
        const v = zero2(f64);
        try expectEqual(v.v, @splat(2, @as(f64, 0)));
    }
    {
        const v = zero3(i8);
        try expectEqual(v.v, @splat(3, @as(i8, 0)));
    }
    {
        const v = zero4(u16);
        try expectEqual(v.v, @splat(4, @as(u16, 0)));
    }
}

pub fn splat2(comptime T: type, val: T) Vec2(T) {
    return Vec2(T){ .v = @splat(2, val) };
}

pub fn splat3(comptime T: type, val: T) Vec3(T) {
    return Vec3(T){ .v = @splat(3, val) };
}

pub fn splat4(comptime T: type, val: T) Vec4(T) {
    return Vec4(T){ .v = @splat(4, val) };
}

test "linear.splat*" {
    {
        const v = splat2(i32, -1);
        try expectEqual(v.v, @splat(2, @as(i32, -1)));
    }
    {
        const v = splat3(f32, 0.25);
        try expectEqual(v.v, @splat(3, @as(f32, 0.25)));
    }
    {
        const v = splat4(u8, 255);
        try expectEqual(v.v, @splat(4, @as(u8, 255)));
    }
}

pub fn identity2x2(comptime T: type) Vec2x2(T) {
    return Vec2x2(T){ .v = .{
        1, 0,
        0, 1,
    } };
}

pub fn identity3x3(comptime T: type) Vec3x3(T) {
    return Vec3x3(T){ .v = .{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    } };
}

pub fn identity4x4(comptime T: type) Vec4x4(T) {
    return Vec4x4(T){ .v = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    } };
}

pub fn identityQuaternion(comptime T: type) Vec4(T) {
    return Vec4(T){ .v = .{ 0, 0, 0, 1 } };
}

test "linear.identity*" {
    {
        const m = identity2x2(f64);
        var n: Vec2x2(f64) = undefined;
        n.identity();
        try expectEqual(m.v, n.v);
        n = .{ .v = .{
            1, 2,
            3, 4,
        } };
        try expectEqual(n.v, n.mul(m).v);
        try expectEqual(n.v, m.mul(n).v);
        try expectEqual(m.v, m.mul(m).v);
    }
    {
        const m = identity3x3(f16);
        var n: Vec3x3(f16) = undefined;
        n.identity();
        try expectEqual(m.v, n.v);
        n = .{ .v = .{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        } };
        try expectEqual(n.v, n.mul(m).v);
        try expectEqual(n.v, m.mul(n).v);
        try expectEqual(m.v, m.mul(m).v);
    }
    {
        const m = identity4x4(f32);
        var n: Vec4x4(f32) = undefined;
        n.identity();
        try expectEqual(m.v, n.v);
        n = .{ .v = .{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
        } };
        try expectEqual(n.v, n.mul(m).v);
        try expectEqual(n.v, m.mul(n).v);
        try expectEqual(m.v, m.mul(m).v);
    }
    {
        const q = identityQuaternion(f32);
        const v = Vec4(f32){ .v = .{ 1, 2, 3, 4 } };
        try expectEqual(v.v, v.mulQuaternion(q).v);
        try expectEqual(v.v, q.mulQuaternion(v).v);
        try expectEqual(q.v, q.mulQuaternion(q).v);
    }
}

pub fn translate4x4(comptime T: type, x: T, y: T, z: T) Vec4x4(T) {
    return Vec4x4(T){ .v = .{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        x, y, z, 1,
    } };
}

pub fn rotate3x3(comptime T: type, axis: Vec3(T), angle: T) Vec3x3(T) {
    return rotateNxN(3, T, axis, angle);
}

pub fn rotate4x4(comptime T: type, axis: Vec3(T), angle: T) Vec4x4(T) {
    return rotateNxN(4, T, axis, angle);
}

fn rotateNxN(comptime n: comptime_int, comptime T: type, axis: Vec3(T), angle: T) VecNxN(n, T) {
    const x_y_z = axis.normalize().v;
    const xx_yy_zz = x_y_z * x_y_z;
    const xy_yz_zx = x_y_z *
        @shuffle(T, x_y_z, undefined, @Vector(3, i32){ 1, 2, 0 });

    const cos = @cos(angle);
    const dcos = 1 - cos;
    const dcosxx_dcosyy_dcoszz = @splat(3, dcos) * xx_yy_zz;
    const dcosxy_dcosyz_dcoszx = @splat(3, dcos) * xy_yz_zx;

    const sin = @sin(angle);
    const sinz_sinx_siny = @splat(3, sin) *
        @shuffle(T, x_y_z, undefined, @Vector(3, i32){ 2, 0, 1 });

    const m00_m11_m22 = @splat(3, cos) + dcosxx_dcosyy_dcoszz;
    const m01_m12_m20 = dcosxy_dcosyz_dcoszx + sinz_sinx_siny;
    const m10_m21_m02 = dcosxy_dcosyz_dcoszx - sinz_sinx_siny;

    switch (n) {
        3 => return Vec3x3(T){ .v = .{
            m00_m11_m22[0],
            m01_m12_m20[0],
            m10_m21_m02[2],

            m10_m21_m02[0],
            m00_m11_m22[1],
            m01_m12_m20[1],

            m01_m12_m20[2],
            m10_m21_m02[1],
            m00_m11_m22[2],
        } },
        4 => return Vec4x4(T){ .v = .{
            m00_m11_m22[0],
            m01_m12_m20[0],
            m10_m21_m02[2],
            0,

            m10_m21_m02[0],
            m00_m11_m22[1],
            m01_m12_m20[1],
            0,

            m01_m12_m20[2],
            m10_m21_m02[1],
            m00_m11_m22[2],
            0,

            0,
            0,
            0,
            1,
        } },
        else => @compileError("rotateNxN requires n == 3 or n == 4"),
    }
}

pub fn rotateQuaternion(comptime T: type, axis: Vec3(T), angle: T) Vec4(T) {
    const norm_axis = axis.normalize().v;
    const half_angle = angle * 0.5;
    // TODO: Maybe use scalar here and construct a @Vector later.
    const cos = @splat(3, @cos(half_angle));
    const sin = @splat(3, @sin(half_angle));
    return Vec4(T){ .v = @shuffle(T, sin * norm_axis, cos, @Vector(4, i32){ 0, 1, 2, -1 }) };
}

pub fn rotate3x3Quaternion(comptime T: type, q: Vec4(T)) Vec3x3(T) {
    return rotateNxNQuaternion(3, T, q);
}

pub fn rotate4x4Quaternion(comptime T: type, q: Vec4(T)) Vec4x4(T) {
    return rotateNxNQuaternion(4, T, q);
}

fn rotateNxNQuaternion(comptime n: comptime_int, comptime T: type, q: Vec4(T)) VecNxN(n, T) {
    const norm_q = q.normalize().v;
    const two = @splat(4, @as(T, 2));

    const xx2_yy2_zz2_ = norm_q * norm_q * two;
    const xy2_yz2_zx2_ = norm_q *
        @shuffle(T, norm_q, undefined, @Vector(4, i32){ 1, 2, 0, 3 }) *
        two;
    const zw2_xw2_yw2_ = @shuffle(
        T,
        norm_q * @splat(4, norm_q[3]) * two,
        undefined,
        @Vector(4, i32){ 2, 0, 1, 3 },
    );

    const m00_m11_m22 = @splat(3, @as(T, 1)) -
        @shuffle(T, xx2_yy2_zz2_, undefined, @Vector(3, i32){ 1, 0, 0 }) -
        @shuffle(T, xx2_yy2_zz2_, undefined, @Vector(3, i32){ 2, 2, 1 });
    const m01_m12_m20_ = xy2_yz2_zx2_ + zw2_xw2_yw2_;
    const m10_m21_m02_ = xy2_yz2_zx2_ - zw2_xw2_yw2_;

    switch (n) {
        3 => return Vec3x3(T){ .v = .{
            m00_m11_m22[0],
            m01_m12_m20_[0],
            m10_m21_m02_[2],

            m10_m21_m02_[0],
            m00_m11_m22[1],
            m01_m12_m20_[1],

            m01_m12_m20_[2],
            m10_m21_m02_[1],
            m00_m11_m22[2],
        } },
        4 => return Vec4x4(T){ .v = .{
            m00_m11_m22[0],
            m01_m12_m20_[0],
            m10_m21_m02_[2],
            0,

            m10_m21_m02_[0],
            m00_m11_m22[1],
            m01_m12_m20_[1],
            0,

            m01_m12_m20_[2],
            m10_m21_m02_[1],
            m00_m11_m22[2],
            0,

            0,
            0,
            0,
            1,
        } },
        else => @compileError("rotateNxNQuaternion requires n == 3 or n == 4"),
    }
}

pub fn scale3x3(comptime T: type, x: T, y: T, z: T) Vec3x3(T) {
    return Vec3x3(T){ .v = .{
        x, 0, 0,
        0, y, 0,
        0, 0, z,
    } };
}

pub fn scale4x4(comptime T: type, x: T, y: T, z: T) Vec4x4(T) {
    return Vec4x4(T){ .v = .{
        x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1,
    } };
}

test "linear.translate*/rotate*/scale*" {
    const m = translate4x4(f32, 2, -4, 3)
        .mul(rotate4x4(f32, Vec3(f32){ .v = .{ 0, 1, 0 } }, 3.141593));
    try expectEqual(@round(m.v), @Vector(16, f32){
        -1, 0,  0,  0,
        0,  1,  0,  0,
        0,  0,  -1, 0,
        2,  -4, 3,  1,
    });
    const n = rotate3x3(f32, Vec3(f32){ .v = .{ 1, 0, 0 } }, 3.141593)
        .mul(scale3x3(f32, 4, 3, 2));
    try expectEqual(@round(n.v), @Vector(9, f32){
        4, 0,  0,
        0, -3, 0,
        0, 0,  -2,
    });
    const o = translate4x4(f32, 2, -4, 3)
        .mul(rotate4x4(f32, Vec3(f32){ .v = .{ 0, 0, 1 } }, 3.141593))
        .mul(scale4x4(f32, 7, 6, 5));
    try expectEqual(@round(o.v), @Vector(16, f32){
        -7, 0,  0, 0,
        0,  -6, 0, 0,
        0,  0,  5, 0,
        2,  -4, 3, 1,
    });
}

test "linear.rotateQuaternion" {
    const pi_4 = 3.141593 / 4.0;
    {
        const rot_x = rotateQuaternion(f32, Vec3(f32){ .v = .{ 1, 0, 0 } }, pi_4);
        const rot_y = rotateQuaternion(f32, Vec3(f32){ .v = .{ 0, 1, 0 } }, pi_4);
        const rot_z = rotateQuaternion(f32, Vec3(f32){ .v = .{ 0, 0, 1 } }, pi_4);
        try expect(rot_x.v[0] > 0);
        try expectEqual(rot_x.v[0], rot_y.v[1]);
        try expectEqual(rot_y.v[1], rot_z.v[2]);
        try expectEqual(rot_x.v[1], 0);
        try expectEqual(rot_x.v[2], 0);
        try expectEqual(rot_y.v[0], 0);
        try expectEqual(rot_y.v[2], 0);
        try expectEqual(rot_z.v[0], 0);
        try expectEqual(rot_z.v[1], 0);
        try expect(rot_x.v[3] > 0);
        try expectEqual(rot_y.v[3], rot_x.v[3]);
        try expectEqual(rot_x.v[3], rot_z.v[3]);
    }
    {
        const rot_x = rotateQuaternion(f32, Vec3(f32){ .v = .{ -1, 0, 0 } }, pi_4);
        const rot_y = rotateQuaternion(f32, Vec3(f32){ .v = .{ 0, -1, 0 } }, pi_4);
        const rot_z = rotateQuaternion(f32, Vec3(f32){ .v = .{ 0, 0, -1 } }, pi_4);
        try expect(rot_x.v[0] < 0);
        try expectEqual(rot_x.v[0], rot_y.v[1]);
        try expectEqual(rot_y.v[1], rot_z.v[2]);
        try expectEqual(rot_x.v[1], 0);
        try expectEqual(rot_x.v[2], 0);
        try expectEqual(rot_y.v[0], 0);
        try expectEqual(rot_y.v[2], 0);
        try expectEqual(rot_z.v[0], 0);
        try expectEqual(rot_z.v[1], 0);
        try expect(rot_x.v[3] > 0);
        try expectEqual(rot_y.v[3], rot_x.v[3]);
        try expectEqual(rot_x.v[3], rot_z.v[3]);
    }
}

test "linear.rotateNxNQuaternion" {
    const angle = 3.141593 / 3.0;
    const axes = [_]Vec3(f32){
        .{ .v = .{ 1, 0, 0 } },
        .{ .v = .{ 0, 1, 0 } },
        .{ .v = .{ 0, 0, 1 } },
        .{ .v = .{ -1, 0, 0 } },
        .{ .v = .{ 0, -1, 0 } },
        .{ .v = .{ 0, 0, -1 } },
        // These functions should not assume that
        // axis is normalized.
        .{ .v = .{ 1, 1, 0 } },
        .{ .v = .{ -2, 2, 0 } },
        .{ .v = .{ 3, 3, -3 } },
        .{ .v = .{ 5, -4, 6 } },
    };
    for (axes) |axis| {
        const q = rotateQuaternion(f32, axis, angle);
        const m = rotate3x3Quaternion(f32, q);
        const n = rotate3x3(f32, axis, angle);
        try expect(@reduce(.Add, @fabs(n.v - m.v)) < 0.000001);
    }
}
