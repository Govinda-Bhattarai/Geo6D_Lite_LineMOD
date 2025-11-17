#!/usr/bin/env python3
"""
Comprehensive verification script to check:
1. All modules are working correctly
2. Modules integrate correctly
3. Dataset loads correctly with config-based paths
4. Model can process dataset samples
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def test_imports():
    """Test 1: Verify all required modules can be imported"""
    print_section("TEST 1: Module Imports")
    errors = []
    
    modules = [
        ("config", "cfg"),
        ("dataset", "LineMODDriveMini"),
        ("models.backbone", "ResNetBackbone"),
        ("models.pose_head", "Geo6DNet"),
        ("models.losses", "geodesic_loss"),
        ("utils.checkpoint", "save_checkpoint"),
        ("utils.metrics", "rotation_error"),
    ]
    
    for module_name, item_name in modules:
        try:
            mod = __import__(module_name, fromlist=[item_name])
            getattr(mod, item_name)
            print_success(f"Imported {module_name}.{item_name}")
        except Exception as e:
            error_msg = f"Failed to import {module_name}.{item_name}: {str(e)}"
            print_error(error_msg)
            errors.append(error_msg)
    
    return len(errors) == 0

def test_config():
    """Test 2: Verify config structure and paths"""
    print_section("TEST 2: Config Structure")
    errors = []
    
    try:
        from config import cfg
        
        # Check config structure
        assert hasattr(cfg, 'BASE_DIR'), "cfg.BASE_DIR missing"
        assert hasattr(cfg, 'get_linemod_paths'), "cfg.get_linemod_paths missing"
        assert hasattr(cfg, 'DEFAULT_CHECKPOINT'), "cfg.DEFAULT_CHECKPOINT missing"
        assert hasattr(cfg, 'DEFAULT_CHECKPOINT_DIR'), "cfg.DEFAULT_CHECKPOINT_DIR missing"
        print_success("Config structure is correct")
        
        # Check BASE_DIR exists
        assert os.path.isdir(cfg.BASE_DIR), f"BASE_DIR does not exist: {cfg.BASE_DIR}"
        print_success(f"BASE_DIR exists: {cfg.BASE_DIR}")
        
        # Test get_linemod_paths function
        test_obj_id = "05"
        paths = cfg.get_linemod_paths(test_obj_id)
        required_keys = ["DATASET_ROOT", "RGB_DIR", "DEPTH_DIR", "MASK_DIR", 
                        "TRAIN_SPLIT", "TEST_SPLIT", "GT_FILE", "INFO_FILE"]
        
        for key in required_keys:
            assert key in paths, f"Missing key in paths: {key}"
            print_success(f"Path '{key}': {paths[key]}")
        
        # Check checkpoint directory can be created
        checkpoint_dir = cfg.DEFAULT_CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        assert os.path.isdir(checkpoint_dir), f"Cannot create checkpoint dir: {checkpoint_dir}"
        print_success(f"Checkpoint directory ready: {checkpoint_dir}")
        
    except Exception as e:
        error_msg = f"Config test failed: {str(e)}"
        print_error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
    
    return len(errors) == 0

def test_dataset_paths():
    """Test 3: Verify dataset paths exist"""
    print_section("TEST 3: Dataset Paths Verification")
    errors = []
    
    try:
        from config import cfg
        
        # Test with object ID "05" (you can change this)
        test_obj_id = "05"
        paths = cfg.get_linemod_paths(test_obj_id)
        
        # Check if dataset root exists
        if not os.path.isdir(paths["DATASET_ROOT"]):
            error_msg = f"Dataset root not found: {paths['DATASET_ROOT']}"
            print_error(error_msg)
            errors.append(error_msg)
        else:
            print_success(f"Dataset root exists: {paths['DATASET_ROOT']}")
        
        # Check required directories
        dirs_to_check = {
            "RGB_DIR": paths["RGB_DIR"],
            "DEPTH_DIR": paths["DEPTH_DIR"],
            "MASK_DIR": paths["MASK_DIR"],
        }
        
        for name, path in dirs_to_check.items():
            if os.path.isdir(path):
                # Count files
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print_success(f"{name} exists with {file_count} files: {path}")
            else:
                error_msg = f"{name} not found: {path}"
                print_warning(error_msg)
                errors.append(error_msg)
        
        # Check required files
        files_to_check = {
            "GT_FILE": paths["GT_FILE"],
            "INFO_FILE": paths["INFO_FILE"],
            "TRAIN_SPLIT": paths["TRAIN_SPLIT"],
            "TEST_SPLIT": paths["TEST_SPLIT"],
        }
        
        for name, path in files_to_check.items():
            if os.path.isfile(path):
                print_success(f"{name} exists: {path}")
            else:
                error_msg = f"{name} not found: {path}"
                print_warning(error_msg)
                errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"Dataset paths test failed: {str(e)}"
        print_error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
    
    return len(errors) == 0

def test_dataset_loading():
    """Test 4: Verify dataset can load samples"""
    print_section("TEST 4: Dataset Loading")
    errors = []
    
    try:
        from dataset import LineMODDriveMini
        import yaml
        
        # Test dataset loading with config paths
        test_obj_id = "05"
        print_info(f"Loading dataset for object ID: {test_obj_id}")
        
        # Try loading a small number of samples
        dataset = LineMODDriveMini(object_ids=[test_obj_id], split="train", max_per_obj=5)
        
        if len(dataset) == 0:
            error_msg = "Dataset loaded but contains 0 samples!"
            print_error(error_msg)
            errors.append(error_msg)
        else:
            print_success(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Try to get a sample
        if len(dataset) > 0:
            try:
                sample = dataset[0]
                
                # Verify sample structure
                required_keys = ["img", "R", "t", "K"]
                for key in required_keys:
                    if key not in sample:
                        error_msg = f"Sample missing key: {key}"
                        print_error(error_msg)
                        errors.append(error_msg)
                    else:
                        print_success(f"Sample contains '{key}' with shape: {sample[key].shape if hasattr(sample[key], 'shape') else type(sample[key])}")
                
                # Check tensor shapes
                assert sample["img"].shape[0] == 3, f"Expected 3 channels, got {sample['img'].shape[0]}"
                assert sample["R"].shape == (3, 3), f"Expected R shape (3,3), got {sample['R'].shape}"
                assert sample["t"].shape == (3,), f"Expected t shape (3,), got {sample['t'].shape}"
                assert sample["K"].shape == (3, 3), f"Expected K shape (3,3), got {sample['K'].shape}"
                
                print_success("Sample structure is correct")
                
            except Exception as e:
                error_msg = f"Failed to load sample: {str(e)}"
                print_error(error_msg)
                traceback.print_exc()
                errors.append(error_msg)
        
        # Test DataLoader integration
        try:
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            print_success(f"DataLoader works! Batch size: {batch['img'].shape[0]}")
        except Exception as e:
            error_msg = f"DataLoader test failed: {str(e)}"
            print_error(error_msg)
            errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"Dataset loading test failed: {str(e)}"
        print_error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
    
    return len(errors) == 0

def test_model_components():
    """Test 5: Verify model components work"""
    print_section("TEST 5: Model Components")
    errors = []
    
    try:
        from models.backbone import ResNetBackbone
        from models.pose_head import Geo6DNet
        from config import cfg
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_info(f"Using device: {device}")
        
        # Test backbone
        try:
            backbone = ResNetBackbone(pretrained=False)
            backbone = backbone.to(device)
            test_input = torch.randn(1, 3, cfg.model.input_res, cfg.model.input_res).to(device)
            features = backbone(test_input)
            print_success(f"Backbone forward pass works! Output shape: {features.shape}")
        except Exception as e:
            error_msg = f"Backbone test failed: {str(e)}"
            print_error(error_msg)
            errors.append(error_msg)
        
        # Test full model
        try:
            model = Geo6DNet(backbone).to(device)
            model.eval()
            
            # Create dummy inputs
            img = torch.randn(1, 3, cfg.model.input_res, cfg.model.input_res).to(device)
            depth = torch.randn(1, 1, cfg.model.input_res, cfg.model.input_res).to(device) if cfg.model.use_depth else None
            K = torch.eye(3).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img, depth, K)
            
            assert "R" in output, "Model output missing 'R'"
            assert "t_off" in output, "Model output missing 't_off'"
            assert output["R"].shape == (1, 3, 3), f"Expected R shape (1,3,3), got {output['R'].shape}"
            assert output["t_off"].shape == (1, 3), f"Expected t_off shape (1,3), got {output['t_off'].shape}"
            
            print_success(f"Model forward pass works! R shape: {output['R'].shape}, t_off shape: {output['t_off'].shape}")
        except Exception as e:
            error_msg = f"Model test failed: {str(e)}"
            print_error(error_msg)
            traceback.print_exc()
            errors.append(error_msg)
        
    except Exception as e:
        error_msg = f"Model components test failed: {str(e)}"
        print_error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
    
    return len(errors) == 0

def test_integration():
    """Test 6: Verify end-to-end integration"""
    print_section("TEST 6: End-to-End Integration")
    errors = []
    
    try:
        from dataset import LineMODDriveMini
        from models.backbone import ResNetBackbone
        from models.pose_head import Geo6DNet
        from models.losses import geodesic_loss, trans_l1_loss
        from torch.utils.data import DataLoader
        from config import cfg
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_info(f"Using device: {device}")
        
        # Load dataset
        test_obj_id = "05"
        dataset = LineMODDriveMini(object_ids=[test_obj_id], split="train", max_per_obj=2)
        
        if len(dataset) == 0:
            error_msg = "Cannot test integration: dataset is empty"
            print_error(error_msg)
            errors.append(error_msg)
            return len(errors) == 0
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Create model
        backbone = ResNetBackbone(pretrained=False)
        model = Geo6DNet(backbone).to(device)
        model.eval()
        
        # Process a batch
        batch = next(iter(loader))
        img = batch["img"].to(device)
        depth = batch["depth"].to(device) if cfg.model.use_depth and batch["depth"] is not None else None
        K = batch["K"].to(device)
        R_gt = batch["R"].to(device)
        t_gt = batch["t"].to(device)
        
        with torch.no_grad():
            output = model(img, depth, K)
        
        # Compute losses
        lossR = geodesic_loss(output["R"], R_gt)
        lossT = trans_l1_loss(output["t_off"], t_gt)
        total_loss = lossR + lossT
        
        print_success(f"End-to-end forward pass works!")
        print_info(f"  Rotation loss: {lossR.item():.4f}")
        print_info(f"  Translation loss: {lossT.item():.4f}")
        print_info(f"  Total loss: {total_loss.item():.4f}")
        
        # Verify output shapes
        assert output["R"].shape == R_gt.shape, "R shape mismatch"
        assert output["t_off"].shape == t_gt.shape, "t_off shape mismatch"
        print_success("Output shapes match ground truth")
        
    except Exception as e:
        error_msg = f"Integration test failed: {str(e)}"
        print_error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
    
    return len(errors) == 0

def test_checkpoint_io():
    """Test 7: Verify checkpoint save/load"""
    print_section("TEST 7: Checkpoint I/O")
    errors = []
    
    try:
        from models.backbone import ResNetBackbone
        from models.pose_head import Geo6DNet
        from utils.checkpoint import save_checkpoint
        from config import cfg
        import tempfile
        import shutil
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model and optimizer
        backbone = ResNetBackbone(pretrained=False)
        model = Geo6DNet(backbone).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Save checkpoint
        test_dir = tempfile.mkdtemp()
        try:
            save_checkpoint(model, optimizer, epoch=0, path=test_dir)
            
            # Check if checkpoint file exists
            checkpoint_files = [f for f in os.listdir(test_dir) if f.endswith('.pth')]
            if len(checkpoint_files) > 0:
                print_success(f"Checkpoint saved: {checkpoint_files[0]}")
                
                # Try loading
                checkpoint_path = os.path.join(test_dir, checkpoint_files[0])
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                
                assert "model_state_dict" in ckpt, "Checkpoint missing model_state_dict"
                assert "optimizer_state_dict" in ckpt, "Checkpoint missing optimizer_state_dict"
                assert "epoch" in ckpt, "Checkpoint missing epoch"
                
                print_success("Checkpoint structure is correct")
                
                # Test loading into new model
                model2 = Geo6DNet(backbone).to(device)
                model2.load_state_dict(ckpt["model_state_dict"])
                print_success("Model state loaded successfully")
            else:
                error_msg = "No checkpoint file created"
                print_error(error_msg)
                errors.append(error_msg)
        finally:
            shutil.rmtree(test_dir)
        
    except Exception as e:
        error_msg = f"Checkpoint I/O test failed: {str(e)}"
        print_error(error_msg)
        traceback.print_exc()
        errors.append(error_msg)
    
    return len(errors) == 0

def main():
    """Run all verification tests"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}🔍 Geo6D-Lite Setup Verification{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Config Structure", test_config),
        ("Dataset Paths", test_dataset_paths),
        ("Dataset Loading", test_dataset_loading),
        ("Model Components", test_model_components),
        ("Integration", test_integration),
        ("Checkpoint I/O", test_checkpoint_io),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_section("SUMMARY")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Colors.BOLD}Results: {passed_count}/{total_count} tests passed{Colors.RESET}\n")
    
    if passed_count == total_count:
        print_success("🎉 All tests passed! Your setup is ready for training.")
        return 0
    else:
        print_warning("⚠️  Some tests failed. Please fix the issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

