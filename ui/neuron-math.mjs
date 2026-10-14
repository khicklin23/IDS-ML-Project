// An educational 2 → 3 → 2 → 2 dense network. No production weights or data.
export const toyExamples = [
  {x:[0,0],target:0}, {x:[0.5,0],target:0}, {x:[0,0.5],target:0},
  {x:[1,0],target:0}, {x:[0,1],target:0}, {x:[0.5,0.5],target:0},
  {x:[1,0.5],target:1}, {x:[0.5,1],target:1}, {x:[1,1],target:1}
];
export function initialNetwork() {
  return [
    {weights:[[0.8,-0.4],[-0.3,0.9],[0.6,0.5]],bias:[0.1,-0.1,-0.2]},
    {weights:[[0.7,-0.4,0.8],[-0.6,0.7,0.3]],bias:[0.05,-0.05]},
    {weights:[[-0.6,-0.2],[0.6,0.2]],bias:[0.1,-0.1]}
  ];
}
export function softmax(logits) {
  const max=Math.max(...logits),exp=logits.map(z=>Math.exp(z-max));
  const sum=exp.reduce((a,b)=>a+b,0);
  return exp.map(e=>e/sum);
}
export function forward(network,inputs) {
  const activations=[inputs.slice()],sums=[],terms=[];
  for(let layer=0;layer<network.length;layer++) {
    const previous=activations[layer],parameters=network[layer];
    const products=parameters.weights.map(weights=>weights.map((w,i)=>w*previous[i]));
    const z=products.map((row,j)=>row.reduce((a,b)=>a+b,parameters.bias[j]));
    terms.push(products);sums.push(z);
    activations.push(layer===network.length-1?softmax(z):z.map(Math.tanh));
  }
  return {activations,sums,terms};
}
export function lossAndGradient(network,examples=toyExamples) {
  const gradient=network.map(layer=>({weights:layer.weights.map(row=>row.map(()=>0)),bias:layer.bias.map(()=>0)}));
  let loss=0;
  for(const {x,target} of examples) {
    const trace=forward(network,x),output=trace.activations.at(-1),logits=trace.sums.at(-1);
    const max=Math.max(...logits);
    loss+=max+Math.log(logits.reduce((sum,z)=>sum+Math.exp(z-max),0))-logits[target];
    let delta=output.map((p,j)=>p-(j===target?1:0));
    for(let layer=network.length-1;layer>=0;layer--) {
      const previous=trace.activations[layer];
      delta.forEach((d,j)=>{
        gradient[layer].bias[j]+=d/examples.length;
        previous.forEach((a,i)=>gradient[layer].weights[j][i]+=d*a/examples.length);
      });
      if(layer>0) delta=previous.map((a,i)=>network[layer].weights.reduce((sum,row,j)=>sum+row[i]*delta[j],0)*(1-a*a));
    }
  }
  return {loss:loss/examples.length,gradient};
}
export function trainStep(network,rate=0.15,examples=toyExamples) {
  const {loss,gradient}=lossAndGradient(network,examples);
  const updated=network.map((layer,k)=>({
    weights:layer.weights.map((row,j)=>row.map((w,i)=>w-rate*gradient[k].weights[j][i])),
    bias:layer.bias.map((b,j)=>b-rate*gradient[k].bias[j])
  }));
  return {network:updated,before:loss,after:lossAndGradient(updated,examples).loss,gradient};
}
