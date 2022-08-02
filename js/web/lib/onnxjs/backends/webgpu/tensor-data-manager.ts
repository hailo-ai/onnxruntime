// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {createView, Tensor} from '../../tensor';
import {createGpuDataManager, GpuDataManager} from './gpu-data-manager';
import {GpuData, GpuDataId, GpuDataType} from './types';

/**
 * manages Tensor ID -> Gpu Data ID
 */
export interface TensorDataManager {
  /**
   * upload a CPU tensor to GPU.
   */
  uploadTensorToGpu(tensor: Tensor, gpuDataType: GpuDataType): GpuData;

  /**
   * create a new GPU tensor.
   */
  createGpuTensor(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): [Tensor, GpuData];

  /**
   * check whether the tensor has GPU data
   */
  hasGpuData(tensorId: Tensor.Id): boolean;

  /**
   * create a reference to the GPU data.
   */
  createGpuRef(tensorId: Tensor.Id, type: Tensor.DataType, dims: readonly number[]): [Tensor, GpuData];

  /**
   * release the GPU resources referred by the tensor.
   */
  releaseGpuTensor(tensorId: Tensor.Id): void;
}

class TensorDataManagerImpl implements TensorDataManager {
  private map: Map<Tensor.Id, GpuDataId>;
  private reverseMap: Map<GpuDataId, Set<Tensor.Id>>;

  constructor(private gpuDataManager: GpuDataManager) {
    this.map = new Map();
    this.reverseMap = new Map();
  }

  private registerIdMapping(tensorId: Tensor.Id, gpuDataId: GpuDataId): void {
    this.map.set(tensorId, gpuDataId);

    let tensorIds = this.reverseMap.get(gpuDataId);
    if (!tensorIds) {
      tensorIds = new Set();
      this.reverseMap.set(gpuDataId, tensorIds);
    }
    tensorIds.add(tensorId);
  }

  uploadTensorToGpu(tensor: Tensor, gpuDataType: GpuDataType): GpuData {
    const gpuDataId = this.map.get(tensor.dataId);
    if (gpuDataId) {
      const gpuData = this.gpuDataManager.get(gpuDataId);
      if (!gpuData) {
        throw new Error('internal error. this should never happen');
      }
      return gpuData;
    }

    const gpuData = this.gpuDataManager.create(tensor.type, tensor.dims, gpuDataType);
    this.registerIdMapping(tensor.dataId, gpuData.id);
    return gpuData;
  }

  createGpuTensor(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): [Tensor, GpuData] {
    const gpuData = this.gpuDataManager.create(type, dims, gpuDataType);
    const tensor = new Tensor(dims, type, undefined, async () => {
      const data = await this.gpuDataManager.download(gpuData.id);
      return createView(data, type);
    });

    this.registerIdMapping(tensor.dataId, gpuData.id);
    return [tensor, gpuData];
  }

  hasGpuData(tensorId: Tensor.Id): boolean {
    return this.map.has(tensorId);
  }

  createGpuRef(tensorId: Tensor.Id, type: Tensor.DataType, dims: readonly number[]): [Tensor, GpuData] {
    const gpuDataId = this.map.get(tensorId);
    if (!gpuDataId) {
      throw new Error('internal error. this should never happen');
    }

    const gpuData = this.gpuDataManager.get(gpuDataId);
    if (!gpuData) {
      throw new Error('internal error. this should never happen');
    }

    const tensor = new Tensor(dims, type, undefined, async () => {
      const data = await this.gpuDataManager.download(gpuData.id);
      return createView(data, type);
    });

    this.registerIdMapping(tensor.dataId, gpuData.id);
    return [tensor, gpuData];
  }

  releaseGpuTensor(tensorId: Tensor.Id): void {
    const gpuDataId = this.map.get(tensorId);
    if (gpuDataId) {
      this.map.delete(tensorId);

      const tensorIds = this.reverseMap.get(gpuDataId);
      if (!tensorIds) {
        throw new Error('internal error. this should never happen');
      }
      tensorIds.delete(tensorId);
      if (tensorIds.size === 0) {
        this.gpuDataManager.release(gpuDataId);
        this.reverseMap.delete(gpuDataId);
      }
    }
  }
}

export const createTensorDataManager = (device: GPUDevice): TensorDataManager =>
    new TensorDataManagerImpl(createGpuDataManager(device));
