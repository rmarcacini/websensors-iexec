pragma solidity ^0.5.1;

import "WebsensorsDoracle.sol";

contract WebsensorsDoracleClient {
    
    // WebsensorsDoracle Address
    address oracleAddr;
    address owner;
    uint8 public sensor_status = 0;
    uint256 sensor_id;
    
    string public status;
    
    constructor (address _oracleAddr) public {
        owner = msg.sender;
        oracleAddr = _oracleAddr;
    }
    
    function myClientFunction(uint256 _sensor_id) public{
        // updating sensor value
        sensor_id = _sensor_id;
        WebsensorsDoracle websensor = WebsensorsDoracle(oracleAddr);
        sensor_status = websensor.sensors_status(sensor_id);

        // Making a decision according to the occurrence or not of an (off-chain) event
        if(sensor_status == uint8(0)){
            // TODO: The sensor has not been updated yet ...
            status = "status 0 - sensor has not been updated yet...";
        }else if(sensor_status == uint8(1)){
            // TODO: Similar events have not happened so far ...
            status = "status 1 - similar events have not happened so far...";
        }else if(sensor_status == uint8(2)){
            // TODO: Similar events have occurred ...
            status = "status 2 - similar events have ocurred...";
        }else{
            // Error. Check for possible off-chain calculation errors ...
            status = "status 5-9 - check off-chain error...";
        }
    }
    
}

