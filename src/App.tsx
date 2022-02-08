import React from 'react';

import { useWebGpu } from './webgpu';
import './App.css';

export const App: React.FC<{}> = () => {
  const canvasRef= useWebGpu()

  return (
    <div className="App">
      <canvas ref={canvasRef} width={window.innerWidth} height={window.innerHeight}/>
    </div>
  );
}

