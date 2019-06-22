pragma solidity ^0.5.1;

import "WebsensorsDoracleInterface.sol";

contract WebsensorsDoracle is WebsensorsDoracleInterface {

    address public owner;
    bytes public last_iexec_result;
    uint256 public last_sensor_id;
    uint8 public last_sensor_status;
    mapping(uint256 => uint8) public sensors_status;
    mapping(address => uint8) public oracles;
    bytes32 public id_iexec;

    constructor(address _owner) public {
        owner = _owner;
    }

    function receiveResult(bytes32 id, bytes calldata result) external {
        if (oracles[msg.sender] != 1) {
            revert("The message sender is not an authorized oracle.");
        }
    
        last_iexec_result = result;
        
        decodeSensor(last_iexec_result);
        
        id_iexec = id;
    }
    
    function addOracle(address oracle) public {
        if(msg.sender != owner){
            revert("The message sender is not the owner.");
        }
        oracles[oracle]=1;
    }
    
    function delOracle(address oracle) public {
        if(msg.sender != owner){
            revert("The message sender is not the owner.");
        }
        delete oracles[oracle];
    }
  
  
    function decodeSensor(bytes memory b) private {

        last_sensor_id = 0;
        for(uint i=1; i < 5; i++){
            uint x = uint8(b[i]);
            uint r = x/16;
            uint s = x%16;
            uint c = r*10+s;
            if(i==1) last_sensor_id += c*1000000;
            if(i==2) last_sensor_id += c*10000;
            if(i==3) last_sensor_id += c*100;
            if(i==4) last_sensor_id += c;
        }
        
        last_sensor_status = uint8(b[5]);
        sensors_status[last_sensor_id] = last_sensor_status;
        
    }
 
  
}
