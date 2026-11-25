module attributes {gpu.container_module} {
  gpu.module @kernels {
    memref.global @mk_handoff : memref<512xi8, 1> = dense<"0x00"> {alignment = 4 : i64}
    memref.global @mk_calls : memref<4xi8, 1> = dense<"0x00"> {alignment = 4 : i64}
    memref.global "private" @smem0 : memref<128xi32, #gpu.address_space<workgroup>> {alignment = 128 : i64}
    memref.global "private" @mk_consumer_smem0 : memref<128xi32, #gpu.address_space<workgroup>> {alignment = 128 : i64}
    gpu.func @mk_producer(%arg_x: memref<128xi32> {llvm.align = 16 : i64}) kernel attributes {nvvm.reqntid = array<i32: 128, 1, 1>} {
      %mk_handoff_global = memref.get_global @mk_handoff : memref<512xi8, 1>
      %arg_mk_handoff = memref.memory_space_cast %mk_handoff_global : memref<512xi8, 1> to memref<512xi8>
      %c1 = arith.constant 0 : index
      %arg_mk_handoff_i32 = memref.view %arg_mk_handoff[%c1][] : memref<512xi8> to memref<128xi32>
      %mk_calls_global = memref.get_global @mk_calls : memref<4xi8, 1>
      %arg_mk_calls = memref.memory_space_cast %mk_calls_global : memref<4xi8, 1> to memref<4xi8>
      %c2 = arith.constant 0 : index
      %arg_mk_calls_i32 = memref.view %arg_mk_calls[%c2][] : memref<4xi8> to memref<1xi32>
      %smem0 = memref.get_global @smem0 : memref<128xi32, #gpu.address_space<workgroup>>
      %s3 = gpu.thread_id x
      %v0 = arith.index_cast %s3 : index to i32
      %i4 = arith.index_cast %v0 : i32 to index
      %v3 = memref.load %arg_x[%i4] : memref<128xi32>
      %v4 = arith.constant 3 : i32
      %v5 = arith.muli %v3, %v4 : i32
      %v6 = arith.constant 1 : i32
      %v7 = arith.addi %v5, %v6 : i32
      %i5 = arith.index_cast %v0 : i32 to index
      memref.store %v7, %smem0[%i5] : memref<128xi32, #gpu.address_space<workgroup>>
      gpu.barrier
      %v8 = arith.constant -1 : i32
      %v9 = arith.muli %v0, %v8 : i32
      %v10 = arith.constant 127 : i32
      %v11 = arith.addi %v9, %v10 : i32
      %i6 = arith.index_cast %v11 : i32 to index
      %v12 = memref.load %smem0[%i6] : memref<128xi32, #gpu.address_space<workgroup>>
      %i7 = arith.index_cast %v0 : i32 to index
      memref.store %v12, %arg_mk_handoff_i32[%i7] : memref<128xi32>
      %v13 = arith.constant 0 : i32
      %v14 = arith.cmpi eq, %v0, %v13 : i32
      scf.if %v14 {
        %i8 = arith.index_cast %v13 : i32 to index
        %v15 = memref.load %arg_mk_calls_i32[%i8] : memref<1xi32>
        %v16 = arith.addi %v15, %v6 : i32
        %i9 = arith.index_cast %v13 : i32 to index
        memref.store %v16, %arg_mk_calls_i32[%i9] : memref<1xi32>
      }
      gpu.return
    }
    gpu.func @mk_consumer(%arg_out: memref<128xi32> {llvm.align = 16 : i64}, %v0: i32) kernel attributes {nvvm.maxntid = array<i32: 256, 1, 1>} {
      %mk_handoff_global = memref.get_global @mk_handoff : memref<512xi8, 1>
      %arg_mk_handoff = memref.memory_space_cast %mk_handoff_global : memref<512xi8, 1> to memref<512xi8>
      %c1 = arith.constant 0 : index
      %arg_mk_handoff_i32 = memref.view %arg_mk_handoff[%c1][] : memref<512xi8> to memref<128xi32>
      %mk_calls_global = memref.get_global @mk_calls : memref<4xi8, 1>
      %arg_mk_calls = memref.memory_space_cast %mk_calls_global : memref<4xi8, 1> to memref<4xi8>
      %c2 = arith.constant 0 : index
      %arg_mk_calls_i32 = memref.view %arg_mk_calls[%c2][] : memref<4xi8> to memref<1xi32>
      %mk_consumer_smem0 = memref.get_global @mk_consumer_smem0 : memref<128xi32, #gpu.address_space<workgroup>>
      %s3 = gpu.thread_id x
      %v1 = arith.index_cast %s3 : index to i32
      %v4 = arith.constant 128 : i32
      %v5 = arith.cmpi slt, %v1, %v4 : i32
      scf.if %v5 {
        %i4 = arith.index_cast %v1 : i32 to index
        %v6 = memref.load %arg_mk_handoff_i32[%i4] : memref<128xi32>
        %v7 = arith.addi %v6, %v0 : i32
        %i5 = arith.index_cast %v1 : i32 to index
        memref.store %v7, %mk_consumer_smem0[%i5] : memref<128xi32, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      scf.if %v5 {
        %i6 = arith.index_cast %v1 : i32 to index
        %v8 = memref.load %mk_consumer_smem0[%i6] : memref<128xi32, #gpu.address_space<workgroup>>
        %v9 = arith.constant 0 : i32
        %i7 = arith.index_cast %v9 : i32 to index
        %v10 = memref.load %arg_mk_calls_i32[%i7] : memref<1xi32>
        %v11 = arith.muli %v8, %v10 : i32
        %v12 = arith.constant 1 : i32
        %v13 = arith.muli %v11, %v12 : i32
        %i8 = arith.index_cast %v1 : i32 to index
        memref.store %v13, %arg_out[%i8] : memref<128xi32>
      }
      gpu.return
    }
  }
}
