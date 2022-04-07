
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path='configs', config_name='config')
def main(config): 
    
    print(OmegaConf.to_object(config))
    
    from lib.run import run
    
    run(config)
    

if __name__ == '__main__':
    main()
